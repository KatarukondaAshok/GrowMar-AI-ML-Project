"""Task 6: reference tables and ratio features for payslips.

Everything here is label-free: the reference is learned from the payslips themselves and the
ratios come from how payroll works (gross = fixed pay for days worked + overtime + variable pay).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OT_HOURS_CAP = 50           # data dictionary: overtime hours are capped at 50
OT_RATE = 2.0               # overtime is paid at 2x the hourly rate
HOURS_PER_MONTH = 26 * 8    # hourly rate = fixed gross / (26 days x 8 hours)
DEDUCTION_PARTS = ["pf_employee", "esi_employee", "professional_tax", "tds", "other_deductions"]

RATIO_FEATURES = [
    "gross_vs_fixed", "gross_vs_expected", "ded_ratio", "net_ratio", "tds_ratio",
    "ded_sum_gap", "net_gap", "ot_hours", "ot_rate_ratio", "fixed_vs_role", "fixed_vs_own", "pf_vs_own",
]


def expected_gross(df: pd.DataFrame, fixed=None) -> pd.Series:
    """What gross pay should be: fixed pay for the days paid + overtime + variable pay."""
    fixed = df["monthly_fixed_gross"] if fixed is None else fixed
    days = pd.to_datetime(df["month"]).dt.days_in_month
    return fixed * df["payable_days"] / days + df["overtime_pay"] + df["variable_pay"]


def fit_reference(df: pd.DataFrame) -> dict:
    """Learn what 'normal' looks like from payslips only (no labels)."""
    d = df.sort_values("payroll_item_id")
    good = d[(d["gross_pay"] > 0) & (d["net_pay"] > 0)]
    net_ratio = good["net_pay"] / good["gross_pay"]
    first_id = d.drop_duplicates(["employee_id", "month"], keep="first")
    return {
        "role_median_fixed": d.groupby("role_id")["monthly_fixed_gross"].median().to_dict(),
        "role_count": d.groupby("role_id").size().to_dict(),
        "emp_median_fixed": d.groupby("employee_id")["monthly_fixed_gross"].median().to_dict(),
        "emp_median_pf": d.groupby("employee_id")["pf_employee"].median().to_dict(),
        "emp_net_ratio": net_ratio.groupby(good["employee_id"]).median().to_dict(),
        "global_net_ratio": float(net_ratio.median()),
        "seen": {f"{e}|{pd.Timestamp(m):%Y-%m}": i for e, m, i in
                 zip(first_id["employee_id"], first_id["month"], first_id["payroll_item_id"])},
    }


def build_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Ratio features that make odd payslips stand out. Index is the same as `df`."""
    gross = df["gross_pay"].astype(float)
    safe_gross = gross.where(gross > 0, np.nan)
    exp = expected_gross(df)
    role_med = df["role_id"].map(ref["role_median_fixed"])
    own_fixed = df["employee_id"].map(ref["emp_median_fixed"])
    own_pf = df["employee_id"].map(ref["emp_median_pf"])
    parts = df[DEDUCTION_PARTS].sum(axis=1)
    hourly = df["monthly_fixed_gross"] / HOURS_PER_MONTH
    ot_expected = df["overtime_hours_paid"] * hourly * OT_RATE
    f = pd.DataFrame(index=df.index)
    f["gross_vs_fixed"] = gross / df["monthly_fixed_gross"]
    f["gross_vs_expected"] = gross / exp.where(exp > 0, np.nan)
    f["ded_ratio"] = df["total_deductions"] / safe_gross
    f["net_ratio"] = df["net_pay"] / safe_gross
    f["tds_ratio"] = df["tds"] / safe_gross
    f["ded_sum_gap"] = (parts - df["total_deductions"]) / safe_gross
    f["net_gap"] = (gross - df["total_deductions"] - df["net_pay"]) / safe_gross
    f["ot_hours"] = df["overtime_hours_paid"].astype(float)
    f["ot_rate_ratio"] = (df["overtime_pay"] / ot_expected.where(ot_expected > 0, np.nan))
    f["fixed_vs_role"] = df["monthly_fixed_gross"] / role_med
    f["fixed_vs_own"] = df["monthly_fixed_gross"] / own_fixed
    f["pf_vs_own"] = (df["pf_employee"] / own_pf.where(own_pf > 0, np.nan))
    # neutral fill: 1.0 for "same as usual" ratios, 0 for gaps, so missing context never looks odd
    neutral = {"gross_vs_fixed": 1.0, "gross_vs_expected": 1.0, "ded_ratio": 0.0, "net_ratio": 1.0, "tds_ratio": 0.0,
               "ded_sum_gap": 0.0, "net_gap": 0.0, "ot_rate_ratio": 1.0, "fixed_vs_role": 1.0,
               "fixed_vs_own": 1.0, "pf_vs_own": 1.0}
    return f.fillna(neutral)[RATIO_FEATURES]
