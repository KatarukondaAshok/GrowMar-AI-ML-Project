"""Task 15: pay-band recommender trained on job-related features only, with a fairness check by group."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
VERSION = "paybands-v1"
NUM = ["job_level", "prior_experience_years", "skills_count", "market_index", "hire_year", "log_band_mid"]
CAT = ["department_id", "work_location", "highest_education"]
# Allowed to explain pay: the job, the band, experience, skills, location and market. NOT allowed: gender, age, career gap.
FEATURES = NUM + CAT


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_band_mid"] = np.log(d["band_mid_2026"])
    d["highest_education"] = d["highest_education"].fillna("Unknown")
    if "offered_annual_fixed" in d:                      # not available when recommending for a NEW offer
        d["ratio"] = d["offered_annual_fixed"] / d["band_mid_2026"]
    return d


def make_model(kind: str, feats_num=NUM, feats_cat=CAT):
    pre = ColumnTransformer([("num", StandardScaler(), feats_num), ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), feats_cat)])
    reg = Ridge(alpha=3.0) if kind == "linear" else LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                                                                  subsample=0.8, subsample_freq=1, reg_lambda=5.0, random_state=SEED, verbose=-1, n_jobs=1)
    return Pipeline([("pre", pre), ("reg", reg)])


class RoleAverage:
    """Baseline from the brief: predict the average offered pay of the role (fallback: job level, then everyone)."""
    def fit(self, d: pd.DataFrame):
        self.role_, self.level_, self.all_ = d.groupby("role_id")["offered_annual_fixed"].mean(), d.groupby("job_level")["offered_annual_fixed"].mean(), d["offered_annual_fixed"].mean()
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return d["role_id"].map(self.role_).fillna(d["job_level"].map(self.level_)).fillna(self.all_).to_numpy()


def pay_metrics(y_true, y_pred) -> dict:
    return {"MAE_rupees": round(float(mean_absolute_error(y_true, y_pred)), 0), "R2": round(float(r2_score(y_true, y_pred)), 4),
            "MAPE_pct": round(float(np.mean(np.abs(y_true - y_pred) / y_true) * 100), 2)}


@dataclass
class PayBandModel:
    """Predicts pay as (band mid-point x ratio) and returns a low / mid / high range."""
    pipeline: Pipeline
    q_low: float          # 10th percentile of out-of-fold log-ratio errors
    q_high: float         # 90th percentile
    features: list
    now_market_index: float = 1.0
    now_year: int = 2026

    def predict_mid(self, df: pd.DataFrame, as_of_now: bool = False) -> np.ndarray:
        d = prepare(df)
        if as_of_now:                                   # a new offer made today: today's market, today's year
            d["market_index"], d["hire_year"] = self.now_market_index, self.now_year
        return np.exp(self.pipeline.predict(d[self.features])) * d["band_mid_2026"].to_numpy()

    def recommend(self, df: pd.DataFrame, as_of_now: bool = True, clip_to_band: bool = True) -> pd.DataFrame:
        """low / mid / high in rupees per year. For a new offer (as_of_now) the range is kept inside band_min..band_max."""
        mid = self.predict_mid(df, as_of_now)
        lo, hi = mid * np.exp(self.q_low), mid * np.exp(self.q_high)
        if clip_to_band:
            lo, hi = np.clip(lo, df["band_min_2026"], df["band_max_2026"]), np.clip(hi, df["band_min_2026"], df["band_max_2026"])
            mid = np.clip(mid, lo, hi)
        return pd.DataFrame({"low": np.round(lo, -2), "mid": np.round(mid, -2), "high": np.round(hi, -2)}, index=df.index)

    def save(self, path: str, metadata: dict):
        joblib.dump({"model": self, "metadata": {"version": VERSION, **metadata}}, path)

    @staticmethod
    def load(path: str) -> "PayBandModel":
        return joblib.load(path)["model"]
