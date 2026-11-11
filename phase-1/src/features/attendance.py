"""Per-employee monthly attendance features (Task 4 uses the same fit/transform style as Task 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

RAW_COLUMNS = [
    "attendance_rate", "unpaid_absence_days", "longest_unpaid_streak_days", "late_arrivals",
    "avg_hours_worked", "overtime_hours", "half_days_unpaid", "paid_leave_days",
]
ENGINEERED = ["present_ratio", "wfh_share", "rate_change_vs_prev", "hours_change_vs_prev",
              "rate_vs_own_mean", "hours_vs_own_mean"]
FEATURE_COLUMNS = RAW_COLUMNS + ENGINEERED
KEY_COLUMNS = ["employee_id", "month"]


class AttendanceFeatureBuilder(BaseEstimator, TransformerMixin):
    """Turns raw monthly attendance rows into model features.

    fit() learns each employee's usual attendance rate and hours (their own baseline).
    A "sudden pattern change" is then a month that is far from that person's own baseline.
    """

    def fit(self, X, y=None):
        g = X.groupby("employee_id")
        self.emp_rate_ = g["attendance_rate"].mean().to_dict()
        self.emp_hours_ = g["avg_hours_worked"].mean().to_dict()
        self.global_rate_ = float(X["attendance_rate"].mean())
        self.global_hours_ = float(X["avg_hours_worked"].mean())
        return self

    def transform(self, X):
        d = X.sort_values(KEY_COLUMNS).copy()
        d["present_ratio"] = d["present_days"] / d["scheduled_days"].clip(lower=1)
        d["wfh_share"] = d["wfh_days"] / d["present_days"].clip(lower=1)
        grp = d.groupby("employee_id")
        d["rate_change_vs_prev"] = grp["attendance_rate"].diff()
        d["hours_change_vs_prev"] = grp["avg_hours_worked"].diff()
        d["rate_vs_own_mean"] = d["attendance_rate"] - d["employee_id"].map(self.emp_rate_).fillna(self.global_rate_)
        d["hours_vs_own_mean"] = d["avg_hours_worked"] - d["employee_id"].map(self.emp_hours_).fillna(self.global_hours_)
        return d.loc[X.index, FEATURE_COLUMNS]  # back to the caller's row order


def build_attendance_pipeline() -> Pipeline:
    """Feature builder + median fill for the first month of each employee (no previous month)."""
    return Pipeline([
        ("build", AttendanceFeatureBuilder()),
        ("impute", SimpleImputer(strategy="median")),
    ])
