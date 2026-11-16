"""Custom scikit-learn steps. Each learns only in fit() so it can be fitted on training data alone."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

from .config import LOCATION_FIXES, SALARY_TYPO_RATIO


class RowCleaner(BaseEstimator, TransformerMixin):
    """Stateless fixes for impossible values (D7, D8, D9, D10, D11, D20). Nothing is learned from data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        low = d["work_location"].astype("string").str.strip().str.lower()
        d["work_location"] = low.map(lambda v: LOCATION_FIXES.get(v, v.title()) if pd.notna(v) else v)
        d["tenure_months"] = d["tenure_months"].clip(lower=0)                       # D7
        d["attendance_rate_3m"] = d["attendance_rate_3m"].clip(upper=1.0)            # D8
        d["wfh_share_3m"] = d["wfh_share_3m"].clip(0.0, 1.0)                         # D10
        exp = d["total_experience_years"].clip(lower=0)                             # D11
        d["total_experience_years"] = np.maximum(exp, d["tenure_months"] / 12)       # D20
        d["highest_education"] = d["highest_education"].astype("string").str.strip().fillna("Unknown")  # M6
        return d


class SalaryTypoFixer(BaseEstimator, TransformerMixin):
    """D6: pay that is ~10x too high. Learns the usual pay-band midpoint of each role in fit()."""

    def __init__(self, ratio: float = SALARY_TYPO_RATIO):
        self.ratio = ratio

    def fit(self, X, y=None):
        implied = X["annual_fixed_pay"] / X["comp_ratio"]
        self.role_mid_ = implied.groupby(X["role_id"]).median().to_dict()
        self.global_mid_ = float(implied.median())
        return self

    def transform(self, X):
        d = X.copy()
        d["annual_fixed_pay"] = d["annual_fixed_pay"].astype(float)  # int column cannot hold pay / 10
        implied = d["annual_fixed_pay"] / d["comp_ratio"]
        ref = d["role_id"].map(self.role_mid_).fillna(self.global_mid_)
        bad = (implied / ref) > self.ratio
        d.loc[bad, "annual_fixed_pay"] = d.loc[bad, "annual_fixed_pay"] / 10
        self.n_fixed_last_ = int(bad.sum())
        return d


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates new features and the 'was missing' flags (M1-M4). Learns department pay medians in fit()."""

    def fit(self, X, y=None):
        self.dept_median_ = X.groupby("department_id")["annual_fixed_pay"].median().to_dict()
        self.global_median_ = float(X["annual_fixed_pay"].median())
        return self

    def transform(self, X):
        d = X.copy()
        # Flags first, while the gaps still exist
        d["never_promoted"] = d["months_since_last_promotion"].isna().astype(int)         # M1
        d["no_raise_yet"] = d["last_raise_pct"].isna().astype(int)                        # M2
        d["not_reviewed_yet"] = d["last_performance_rating"].isna().astype(int)           # M3
        d["engagement_missing"] = d["last_engagement_score"].isna().astype(int)           # M4
        # Fill "not applicable" gaps with meaningful values
        d["months_since_last_promotion"] = d["months_since_last_promotion"].fillna(d["tenure_months"])
        d["last_raise_pct"] = d["last_raise_pct"].fillna(0.0)
        # New numbers
        d["log_pay"] = np.log10(d["annual_fixed_pay"])
        med = d["department_id"].map(self.dept_median_).fillna(self.global_median_)
        d["pay_vs_dept_median"] = d["annual_fixed_pay"] / med
        d["tenure_years"] = d["tenure_months"] / 12
        d["attendance_gap"] = 1 - d["attendance_rate_3m"]
        d["overtime_per_year_of_tenure"] = d["avg_overtime_hours_per_month_3m"] / np.maximum(d["tenure_years"], 0.5)
        return d


class Winsorizer(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """D3: cap each column at its 1st / 99th percentile (learned on training data only)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)  # NaN stays NaN
