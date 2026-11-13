"""Task 13: attrition model v2 (gradient boosting) compared honestly with the Task 3 baseline on the SAME folds."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.features.config import RANDOM_SEED, TARGET
from src.features.data import load_clean_base, split_train_test
from src.features.pipeline import build_feature_pipeline, output_columns

warnings.filterwarnings("ignore")
VERSION = "attrition-v2"
FLAG_SHARE = 0.30        # High + Medium bands = the top 30% of scores


def folds(n_splits: int = 5):
    """Exactly the Task 3 folds (stratified, shuffled, seed 42)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


def lgbm(**p) -> LGBMClassifier:
    base = dict(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=5.0, random_state=RANDOM_SEED, verbose=-1, n_jobs=1)
    base.update(p)
    return LGBMClassifier(**base)


def baseline_lr() -> LogisticRegression:
    return LogisticRegression(max_iter=3000, class_weight="balanced")


def load_train_test(path: str):
    df = load_clean_base(path)
    train, test = split_train_test(df)            # the same locked split as Task 3
    return train.drop(columns=[TARGET]), train[TARGET], test.drop(columns=[TARGET]), test[TARGET]


def top_share_metrics(y, p, share: float = FLAG_SHARE) -> dict:
    """Flag the riskiest `share` of people. Precision = flagged who really left; recall = leavers caught."""
    y = np.asarray(y); n = max(int(round(len(y) * share)), 1)
    idx = np.argsort(-np.asarray(p), kind="stable")[:n]
    caught = y[idx].sum()
    return {"precision_top30": caught / n, "recall_top30": caught / max(y.sum(), 1)}


def score_all(y, p) -> dict:
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), **top_share_metrics(y, p),
            "brier": brier_score_loss(y, np.clip(p, 0, 1))}


def cv_oof(model, Xtr: pd.DataFrame, ytr: pd.Series, n_splits: int = 5):
    """Out-of-fold probabilities and a per-fold metric table. The feature pipeline is refit inside every fold."""
    oof = np.zeros(len(Xtr)); rows = []
    for k, (a, b) in enumerate(folds(n_splits).split(Xtr, ytr)):
        pipe = Pipeline([("features", build_feature_pipeline()), ("model", model.__class__(**model.get_params()))])
        pipe.fit(Xtr.iloc[a], ytr.iloc[a])
        p = pipe.predict_proba(Xtr.iloc[b])[:, 1]; oof[b] = p
        rows.append({"fold": k + 1, **score_all(ytr.iloc[b], p)})
    return oof, pd.DataFrame(rows)


@dataclass
class AttritionModel:
    """Feature pipeline + gradient boosting + Platt calibration, saved as one versioned object."""
    pipeline: object
    booster: LGBMClassifier
    calibrator: LogisticRegression
    feature_names: list

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.pipeline.transform(X), columns=self.feature_names, index=X.index)

    def raw_logit(self, Xt: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.booster.predict_proba(Xt)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated 12-month leaving probability."""
        return self.calibrator.predict_proba(self.raw_logit(self.transform(X)).reshape(-1, 1))[:, 1]

    def save(self, path: str, metadata: dict):
        joblib.dump({"model": self, "metadata": {"version": VERSION, **metadata}}, path)

    @staticmethod
    def load(path: str) -> "AttritionModel":
        return joblib.load(path)["model"]


def fit_final(Xtr, ytr, params: dict, oof_raw: np.ndarray) -> AttritionModel:
    """Fit on all training rows; Platt-calibrate on the out-of-fold scores (no leakage into the calibrator)."""
    pipe = build_feature_pipeline().fit(Xtr)
    names = output_columns(pipe)
    Xt = pd.DataFrame(pipe.transform(Xtr), columns=names, index=Xtr.index)
    booster = lgbm(**params).fit(Xt, ytr)
    p = np.clip(oof_raw, 1e-6, 1 - 1e-6)
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p)).reshape(-1, 1), ytr)
    return AttritionModel(pipe, booster, cal, names)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p); edges = np.quantile(p, np.linspace(0, 1, bins + 1)); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
