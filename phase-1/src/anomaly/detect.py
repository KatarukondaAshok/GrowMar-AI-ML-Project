"""Isolation Forest detector, threshold sweep and save/load with metadata."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline

from src.features.attendance import FEATURE_COLUMNS, build_attendance_pipeline

SEED = 42
VERSION = "attendance-anomaly-1.0.0"


def make_detector(n_estimators: int = 300, seed: int = SEED) -> Pipeline:
    feats = build_attendance_pipeline()
    return Pipeline(feats.steps + [("forest", IsolationForest(n_estimators=n_estimators, contamination="auto",
                                                              random_state=seed, n_jobs=1))])


def anomaly_scores(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Higher = more unusual (the negative of sklearn's score_samples)."""
    X = pipe[:-1].transform(df)
    return -pipe.named_steps["forest"].score_samples(X)


def sweep_thresholds(scores: np.ndarray, is_anomaly: np.ndarray, flag_rates=None) -> pd.DataFrame:
    """For each share of rows flagged: catch rate (recall) on injected, false-alarm rate on normal rows, precision."""
    flag_rates = flag_rates if flag_rates is not None else [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
                                                           0.05, 0.06, 0.08, 0.10, 0.15, 0.20]
    is_anomaly = np.asarray(is_anomaly).astype(bool)
    rows = []
    for r in flag_rates:
        thr = float(np.quantile(scores, 1 - r))
        flagged = scores >= thr
        tp = int((flagged & is_anomaly).sum()); fp = int((flagged & ~is_anomaly).sum())
        n_pos, n_neg = int(is_anomaly.sum()), int((~is_anomaly).sum())
        prec = tp / max(tp + fp, 1); rec = tp / n_pos
        rows.append({"flag_rate": r, "threshold": thr, "flagged": int(flagged.sum()), "caught": tp, "missed": n_pos - tp,
                     "false_alarms": fp, "catch_rate": rec, "false_alarm_rate": fp / n_neg, "precision": prec,
                     "f1": 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)})
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame, max_flag_rate: float = 0.05) -> pd.Series:
    """Best F1 among thresholds that flag at most `max_flag_rate` of rows (so HR keeps trusting the list)."""
    ok = sweep[sweep["flag_rate"] <= max_flag_rate]
    return ok.loc[ok["f1"].idxmax()]


def save_detector(pipe: Pipeline, threshold: float, out_dir: str, extra: dict | None = None) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "random_seed": SEED, "sklearn_version": sklearn.__version__, "model": "IsolationForest",
            "n_estimators": pipe.named_steps["forest"].n_estimators, "features": FEATURE_COLUMNS,
            "threshold_on_anomaly_score": float(threshold),
            "rule": "flag a month when anomaly_score >= threshold (score = -IsolationForest.score_samples)"}
    meta.update(extra or {})
    path = out / f"{VERSION}.joblib"
    joblib.dump({"pipeline": pipe, "threshold": float(threshold), "metadata": meta}, path)
    (out / f"{VERSION}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return path


def load_detector(path: str):
    b = joblib.load(path)
    return b["pipeline"], b["threshold"], b["metadata"]
