"""Task 6: readable rule checks. Finance teams trust a rule they can read more than a model they cannot."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.payroll.features import OT_HOURS_CAP, DEDUCTION_PARTS, expected_gross

# Order = priority when one payslip breaks several rules (most specific / most expensive first).
RULE_PRIORITY = ["DUPLICATE_PAYMENT", "ZERO_PAY", "NEGATIVE_NET_PAY", "GROSS_ABOVE_NORM", "GROSS_BELOW_EXPECTED",
                 "OVERTIME_ABOVE_CAP", "PF_NOT_DEDUCTED", "ROLE_PAY_ABOVE_PEERS", "NET_PAY_MISMATCH"]

GROSS_ABOVE_LIMIT = 1.5      # gross more than 1.5x what days worked allow
GROSS_BELOW_LIMIT = 0.95     # gross more than 5% under what days worked allow
ROLE_PEER_LIMIT = 2.0        # fixed pay more than 2x the median of the same role
MIN_ROLE_PEERS = 5           # do not judge a role with fewer payslips than this
ARITHMETIC_TOLERANCE = 5.0   # rupees; pay arithmetic should match to the rupee


def apply_rules(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """One True/False column per rule, same index as df."""
    exp = expected_gross(df)
    ratio = df["gross_pay"] / exp.where(exp > 0, np.nan)
    parts = df[DEDUCTION_PARTS].sum(axis=1)
    role_med = df["role_id"].map(ref["role_median_fixed"])
    role_n = df["role_id"].map(ref["role_count"]).fillna(0)
    own_pf = df["employee_id"].map(ref["emp_median_pf"]).fillna(0)
    key = df["employee_id"].astype(str) + "|" + pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    first_id = key.map(ref["seen"])
    out = pd.DataFrame(index=df.index)
    # A payslip is a duplicate when another payslip with a lower id already paid that employee-month.
    out["DUPLICATE_PAYMENT"] = first_id.notna() & (first_id != df["payroll_item_id"]) & \
        (first_id.fillna("") < df["payroll_item_id"])
    out["ZERO_PAY"] = (df["gross_pay"] == 0) & (df["payable_days"] > 0)
    out["NEGATIVE_NET_PAY"] = df["net_pay"] < 0
    out["GROSS_ABOVE_NORM"] = ratio > GROSS_ABOVE_LIMIT
    out["GROSS_BELOW_EXPECTED"] = (ratio > 0) & (ratio < GROSS_BELOW_LIMIT)
    out["OVERTIME_ABOVE_CAP"] = df["overtime_hours_paid"] > OT_HOURS_CAP
    out["PF_NOT_DEDUCTED"] = (df["pf_employee"] == 0) & (own_pf > 0) & (df["gross_pay"] > 0)
    out["ROLE_PAY_ABOVE_PEERS"] = (df["monthly_fixed_gross"] / role_med > ROLE_PEER_LIMIT) & (role_n >= MIN_ROLE_PEERS)
    out["NET_PAY_MISMATCH"] = ((df["gross_pay"] - df["total_deductions"] - df["net_pay"]).abs() > ARITHMETIC_TOLERANCE) | \
        ((parts - df["total_deductions"]).abs() > ARITHMETIC_TOLERANCE)
    return out[RULE_PRIORITY]


def first_rule(hits: pd.DataFrame) -> pd.Series:
    """Highest-priority rule that fired for each row (None if no rule fired)."""
    arr = hits.to_numpy()
    codes = np.array(hits.columns, dtype=object)
    idx = arr.argmax(axis=1)
    return pd.Series(np.where(arr.any(axis=1), codes[idx], None), index=hits.index, dtype=object)
