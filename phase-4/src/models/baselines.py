"""Task 3 helpers: baseline models that plug into the Task 2 feature pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features.config import RANDOM_SEED, TARGET
from src.features.pipeline import build_feature_pipeline

SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
}


def model_zoo() -> dict:
    """The five baselines compared in Task 3."""
    return {
        "Dummy (always stays)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Logistic Regression (balanced)": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Decision Tree depth 4": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=RANDOM_SEED),
        "Decision Tree depth 4 (balanced)": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_SEED),
    }


def with_features(model) -> Pipeline:
    """Feature pipeline + model in ONE object, so cross-validation refits the preprocessing inside every fold."""
    return Pipeline([("features", build_feature_pipeline()), ("model", model)])


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]


def make_cv(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


def cross_validate_all(X: pd.DataFrame, y: pd.Series, models: dict | None = None):
    """Return (summary table of mean and std per metric, per-fold table)."""
    models = models or model_zoo()
    rows, folds = [], []
    for name, est in models.items():
        res = cross_validate(with_features(est), X, y, cv=make_cv(), scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for m in SCORING:
            vals = res[f"test_{m}"]
            row[f"{m}_mean"], row[f"{m}_std"] = float(vals.mean()), float(vals.std())
            for i, v in enumerate(vals):
                folds.append({"model": name, "fold": i + 1, "metric": m, "value": float(v)})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(folds)


def lift_table(y_true: np.ndarray, proba: np.ndarray, fractions=(0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    """If HR reviews only the top X% highest-risk people, how many real leavers are found?"""
    order = np.argsort(-proba)
    y_sorted = np.asarray(y_true)[order]
    total = y_sorted.sum()
    out = []
    for f in fractions:
        k = int(round(f * len(y_sorted)))
        caught = int(y_sorted[:k].sum())
        out.append({"review_top_pct": int(f * 100), "people_reviewed": k, "leavers_found": caught,
                    "share_of_all_leavers_found": caught / total, "precision_in_group": caught / k})
    return pd.DataFrame(out)
