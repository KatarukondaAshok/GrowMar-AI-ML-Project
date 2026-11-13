"""Shared fairness helpers (Tasks 12, 14, 15). Written here because the starter file cortexhr_helpers.py was not supplied."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_gap(df: pd.DataFrame, value_col: str, group_col: str, reference: str | None = None) -> pd.DataFrame:
    """Mean of `value_col` per group and the gap to a reference group (default: the largest group)."""
    g = df.groupby(group_col)[value_col].agg(["mean", "count"]).rename(columns={"mean": "mean_value"})
    ref = reference if reference in g.index else g["count"].idxmax()
    g["gap_vs_" + str(ref)] = g["mean_value"] - g.loc[ref, "mean_value"]
    return g.round(4).reset_index()


def risk_bands(scores, high_share: float = 0.10, medium_share: float = 0.20) -> pd.Series:
    """High = top 10% of scores, Medium = next 20%, Low = the rest (rule stated in sample_attrition_risk)."""
    s = pd.Series(np.asarray(scores, dtype=float))
    rank = s.rank(method="first", ascending=False)
    n = len(s)
    n_high = int(round(n * high_share)); n_med = int(round(n * medium_share))
    return pd.Series(np.where(rank <= n_high, "High", np.where(rank <= n_high + n_med, "Medium", "Low")), index=s.index)
