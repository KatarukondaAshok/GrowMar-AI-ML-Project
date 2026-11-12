"""Reusable data-audit helpers for the CortexHR attrition dataset (Task 1).

Every function takes a DataFrame and returns a small DataFrame/Series so the
notebook, the tests and the report all use the same numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TARGET = "left_within_12m"

# Columns that only exist AFTER the employee has left (blank for everyone who stayed).
POST_EXIT_COLUMNS = [
    "exit_date",
    "resignation_date",
    "exit_reason",
    "exit_interview_done",
    "fnf_settlement_amount",
]
# Derived directly from the label, so it is the answer in disguise.
LABEL_DERIVED_COLUMNS = ["voluntary_left_within_12m"]
# Identifiers / bookkeeping, not real signal.
ID_COLUMNS = ["row_id", "employee_id", "source", "observation_date"]


def load_attrition(path: str, sheet: str = "attrition_dataset") -> pd.DataFrame:
    """Read the attrition sheet of the CortexHR workbook."""
    return pd.read_excel(path, sheet_name=sheet)


def find_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that repeat another row in every column except the row counter."""
    cols = [c for c in df.columns if c != "row_id"]
    return df[df.duplicated(subset=cols, keep=False)].sort_values("employee_id")


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != "row_id"]
    return df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and percent per column (only columns with gaps)."""
    n = df.isna().sum()
    out = pd.DataFrame({"missing": n, "percent": (n / len(df) * 100).round(2)})
    return out[out["missing"] > 0].sort_values("percent", ascending=False)


def structural_missing_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Test whether a gap is 'not applicable' (structural) instead of 'lost data'."""
    rows = []
    # No promotion yet -> no 'months since promotion'
    rows.append(
        (
            "months_since_last_promotion",
            "promotions_received == 0",
            int(((df.promotions_received == 0) & df.months_since_last_promotion.isna()).sum()),
            int(df.months_since_last_promotion.isna().sum()),
        )
    )
    # No raise yet -> no 'last raise pct'
    rows.append(
        (
            "last_raise_pct",
            "raises_received == 0",
            int(((df.raises_received == 0) & df.last_raise_pct.isna()).sum()),
            int(df.last_raise_pct.isna().sum()),
        )
    )
    # No review yet for very new joiners
    new = df.tenure_months <= 12
    rows.append(
        (
            "last_performance_rating",
            "tenure_months <= 12",
            int((new & df.last_performance_rating.isna()).sum()),
            int(df.last_performance_rating.isna().sum()),
        )
    )
    return pd.DataFrame(
        rows, columns=["column", "explained_by", "gaps_explained", "total_gaps"]
    )


def missing_vs_target(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Attrition rate for rows where the value is present vs missing."""
    out = []
    for c in cols:
        grp = df.groupby(df[c].isna())[TARGET].mean()
        out.append(
            {
                "column": c,
                "attrition_when_present": round(grp.get(False, np.nan), 3),
                "attrition_when_missing": round(grp.get(True, np.nan), 3),
            }
        )
    return pd.DataFrame(out)


def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def z_outliers(s: pd.Series, thr: float = 3.0) -> pd.Series:
    s = s.dropna()
    return pd.Series(np.abs(stats.zscore(s)) > thr, index=s.index)


def outlier_compare(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Count outliers with both methods and report skew."""
    rows = []
    for c in cols:
        s = df[c]
        rows.append(
            {
                "column": c,
                "iqr_outliers": int(iqr_outliers(s).sum()),
                "zscore_outliers": int(z_outliers(s).sum()),
                "skew": round(float(s.skew()), 2),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def salary_typos(df: pd.DataFrame, factor: float = 2.0) -> pd.DataFrame:
    """Pay that is wildly out of line with the pay band implied by comp_ratio.

    comp_ratio = pay / role midpoint, so pay / comp_ratio should be about the same
    for every person in the same role. If it is ~10x the role median, pay is mistyped.
    """
    d = df.copy()
    d["implied_midpoint"] = d.annual_fixed_pay / d.comp_ratio
    role_median = d.groupby("role_id")["implied_midpoint"].transform("median")
    d["midpoint_vs_role_median"] = (d.implied_midpoint / role_median).round(2)
    bad = d[
        (d.midpoint_vs_role_median > factor) | (d.midpoint_vs_role_median < 1 / factor)
    ]
    return bad[
        ["employee_id", "role_id", "job_level", "annual_fixed_pay", "comp_ratio", "midpoint_vs_role_median"]
    ]


def impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Rule checks for values that cannot be true."""
    rules = {
        "tenure_months < 0": df.tenure_months < 0,
        "total_experience_years < 0": df.total_experience_years < 0,
        "attendance_rate_3m > 1": df.attendance_rate_3m > 1,
        "wfh_share_3m < 0": df.wfh_share_3m < 0,
        "wfh_share_3m > 1": df.wfh_share_3m > 1,
        "tenure longer than total experience": (df.tenure_months / 12) > df.total_experience_years + 0.5,
        "avg_hours_worked_3m < 4": df.avg_hours_worked_3m < 4,
    }
    return pd.DataFrame(
        {"rule": list(rules), "rows_failing": [int(v.sum()) for v in rules.values()]}
    )


def messy_categories(df: pd.DataFrame, col: str = "work_location") -> pd.DataFrame:
    """Spellings that collapse to the same place after cleaning."""
    canon = {"bangalore": "Bengaluru", "madras": "Chennai", "delhi-ncr": "Delhi NCR"}
    s = df[col].str.strip()
    key = s.str.lower().map(lambda x: canon.get(x, x.title() if x != "delhi ncr" else "Delhi NCR"))
    changed = df.loc[s != key, [col]].assign(cleaned=key[s != key])
    return changed.value_counts().rename("rows").reset_index()


def clean_location(s: pd.Series) -> pd.Series:
    canon = {"bangalore": "Bengaluru", "madras": "Chennai", "delhi-ncr": "Delhi NCR", "delhi ncr": "Delhi NCR"}
    low = s.str.strip().str.lower()
    return low.map(lambda x: canon.get(x, x.title()))


def class_balance(df: pd.DataFrame) -> dict:
    n_left = int(df[TARGET].sum())
    n_stay = int(len(df) - n_left)
    return {
        "rows": len(df),
        "left": n_left,
        "stayed": n_stay,
        "left_pct": round(n_left / len(df) * 100, 2),
        "ratio_stay_to_left": round(n_stay / n_left, 2),
        "dummy_accuracy_pct": round(n_stay / len(df) * 100, 2),
    }


def leakage_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each suspect column: how often is it filled for leavers vs stayers?"""
    rows = []
    for c in POST_EXIT_COLUMNS:
        rows.append(
            {
                "column": c,
                "filled_for_leavers_pct": round(df.loc[df[TARGET] == 1, c].notna().mean() * 100, 1),
                "filled_for_stayers_pct": round(df.loc[df[TARGET] == 0, c].notna().mean() * 100, 1),
                "verdict": "LEAKS (only known after exit)",
            }
        )
    for c in LABEL_DERIVED_COLUMNS:
        agree = ((df[c] == 1) <= (df[TARGET] == 1)).mean() * 100
        rows.append(
            {
                "column": c,
                "filled_for_leavers_pct": np.nan,
                "filled_for_stayers_pct": np.nan,
                "verdict": f"LEAKS (subset of the label, consistent in {agree:.0f}% of rows)",
            }
        )
    return pd.DataFrame(rows)


def single_feature_auc(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """How well does one column alone separate leavers from stayers (0.5 = no signal)?

    A value near 1.0 for an ordinary feature is a red flag for leakage.
    """
    from sklearn.metrics import roc_auc_score

    out = {}
    for c in features:
        s = df[c]
        if s.nunique(dropna=True) < 2:
            continue
        s = s.fillna(s.median())
        a = roc_auc_score(df[TARGET], s)
        out[c] = max(a, 1 - a)  # direction does not matter for this check
    return pd.Series(out).sort_values(ascending=False).round(3)
