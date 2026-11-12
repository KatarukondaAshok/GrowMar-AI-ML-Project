"""The reusable feature pipeline: build, fit, save, load, transform."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import FEATURE_VERSION
from .config import (CATEGORICAL, FLAG_COLUMNS, ID_COLUMNS, LEAKAGE_COLUMNS, NUMERIC_BASE,
                     NUMERIC_COLUMNS, RANDOM_SEED)
from .transformers import FeatureEngineer, RowCleaner, SalaryTypoFixer, Winsorizer


def build_feature_pipeline() -> Pipeline:
    """Return an UNFITTED pipeline. Cloneable, so cross-validation can refit it inside every fold."""
    numeric = Pipeline([
        ("cap", Winsorizer(0.01, 0.99)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])
    encode = ColumnTransformer(
        [("num", numeric, NUMERIC_COLUMNS),
         ("flag", "passthrough", FLAG_COLUMNS),
         ("cat", categorical, CATEGORICAL)],
        remainder="drop",              # leakage, identifiers and everything else are dropped here
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("clean", RowCleaner()),
        ("fix_salary", SalaryTypoFixer()),
        ("engineer", FeatureEngineer()),
        ("encode", encode),
    ])


def fit_pipeline(train_df: pd.DataFrame) -> Pipeline:
    """Fit on TRAINING rows only. Never pass the test set here."""
    pipe = build_feature_pipeline()
    pipe.fit(train_df)
    return pipe


def output_columns(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["encode"].get_feature_names_out())


def transform_dataframe(pipe: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(pipe.transform(df), columns=output_columns(pipe), index=df.index)


def make_metadata(pipe: Pipeline, train_df: pd.DataFrame, extra: dict | None = None) -> dict:
    cleaned = pipe.named_steps["clean"].transform(train_df)
    pipe.named_steps["fix_salary"].transform(cleaned)  # sets n_fixed_last_
    meta = {
        "version": FEATURE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "n_training_rows": int(len(train_df)),
        "required_input_columns": sorted(set(NUMERIC_BASE) | set(CATEGORICAL) | {"role_id", "annual_fixed_pay"}),
        "dropped_leakage_columns": LEAKAGE_COLUMNS,
        "dropped_id_columns": ID_COLUMNS,
        "salary_typos_fixed_in_training": int(pipe.named_steps["fix_salary"].n_fixed_last_),
        "n_output_columns": len(output_columns(pipe)),
        "output_columns": output_columns(pipe),
    }
    meta.update(extra or {})
    return meta


def save_pipeline(pipe: Pipeline, train_df: pd.DataFrame, out_dir: str, extra: dict | None = None) -> Path:
    """Save the fitted pipeline and its metadata (version, columns, ...)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = make_metadata(pipe, train_df, extra)
    path = out / f"{FEATURE_VERSION}.joblib"
    joblib.dump({"pipeline": pipe, "metadata": meta}, path)
    (out / f"{FEATURE_VERSION}.meta.json").write_text(json.dumps(meta, indent=2))
    return path


def load_pipeline(path: str):
    """Return (pipeline, metadata) saved by save_pipeline."""
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["metadata"]
