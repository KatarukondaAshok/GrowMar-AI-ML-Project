"""Loading, de-duplicating and splitting the attrition table (done BEFORE any fitting)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATE_COLUMN, DEFAULT_SHEET, RANDOM_SEED, TARGET


def load_raw(path: str, sheet: str = DEFAULT_SHEET) -> pd.DataFrame:
    """Read the attrition sheet exactly as exported (dirty)."""
    return pd.read_excel(path, sheet_name=sheet)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """D1: remove exact duplicate rows (ignoring the row counter)."""
    cols = [c for c in df.columns if c != "row_id"]
    return df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)


def load_clean_base(path: str, sheet: str = DEFAULT_SHEET) -> pd.DataFrame:
    return drop_duplicates(load_raw(path, sheet))


def split_train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = RANDOM_SEED):
    """Stratified split. The returned test part is LOCKED: do not look at it until the final check."""
    return train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)


def time_based_split(df: pd.DataFrame, cutoff: str = "2025-01-01"):
    """Train on older observation dates, test on the newest. Used as a robustness check (D18)."""
    older = df[pd.to_datetime(df[DATE_COLUMN]) < pd.Timestamp(cutoff)]
    newer = df[pd.to_datetime(df[DATE_COLUMN]) >= pd.Timestamp(cutoff)]
    return older, newer
