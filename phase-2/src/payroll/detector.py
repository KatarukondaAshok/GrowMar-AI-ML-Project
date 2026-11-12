"""Task 6: rules + Isolation Forest, trained without labels, saved with a version."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from src.payroll.features import RATIO_FEATURES, build_features, fit_reference
from src.payroll.rules import RULE_PRIORITY, apply_rules, first_rule

SEED = 42
VERSION = "payroll-anomaly-1.0.0"
CONTAMINATION = 0.025      # same flag budget as the reference baseline (2.5% of payslips); chosen without labels


class PayrollDetector:
    """Readable rules first, Isolation Forest on the ratio features for what the rules cannot see."""

    def __init__(self, contamination: float = CONTAMINATION, n_estimators: int = 300, seed: int = SEED):
        self.contamination, self.n_estimators, self.seed = contamination, n_estimators, seed

    def fit(self, df: pd.DataFrame) -> "PayrollDetector":
        self.ref_ = fit_reference(df)
        X = build_features(df, self.ref_)
        self.forest_ = IsolationForest(n_estimators=self.n_estimators, contamination="auto",
                                       random_state=self.seed, n_jobs=1).fit(X)
        raw = -self.forest_.score_samples(X)
        self.score_min_, self.score_max_ = float(raw.min()), float(raw.max())
        self.threshold_ = float(np.quantile(raw, 1 - self.contamination))   # flag the top 2.5%
        return self

    def raw_scores(self, df: pd.DataFrame) -> np.ndarray:
        return -self.forest_.score_samples(build_features(df, self.ref_))

    def scale(self, raw: np.ndarray) -> np.ndarray:
        """0 to 1 anomaly score (1 = most unusual payslip seen in training)."""
        span = max(self.score_max_ - self.score_min_, 1e-9)
        return np.clip((raw - self.score_min_) / span, 0, 1)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per payslip: model score, rule hits and the final flag (rules OR model)."""
        raw = self.raw_scores(df)
        hits = apply_rules(df, self.ref_)
        out = pd.DataFrame(index=df.index)
        out["model_raw"] = raw
        out["model_score"] = self.scale(raw)
        out["model_flag"] = raw >= self.threshold_
        out["rule_flag"] = hits.any(axis=1)
        out["rule_code"] = first_rule(hits)
        out["flagged"] = out["rule_flag"] | out["model_flag"]
        out["anomaly_score"] = np.where(out["rule_flag"], 1.0, out["model_score"]).round(3)
        return pd.concat([out, hits.add_prefix("rule_")], axis=1)

    def save(self, out_dir: str) -> Path:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        meta = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "random_seed": self.seed, "sklearn_version": sklearn.__version__, "model": "IsolationForest + rules",
                "n_estimators": self.n_estimators, "features": RATIO_FEATURES, "rules": RULE_PRIORITY,
                "contamination": self.contamination, "threshold_on_raw_score": self.threshold_,
                "trained_on_rows": int(len(self.ref_["seen"])), "labels_used_for_training": False,
                "flag_rule": "flagged = any rule fires OR raw IsolationForest score >= threshold"}
        path = out / f"{VERSION}.joblib"
        joblib.dump({"detector": self, "metadata": meta}, path)
        (out / f"{VERSION}.meta.json").write_text(json.dumps(meta, indent=2))
        return path

    @staticmethod
    def load(path: str) -> "PayrollDetector":
        return joblib.load(path)["detector"]
