"""Task 7: turn a flag into a reason code, a sentence a clerk understands and the money at risk.

Official columns come from the sheet `sample_anomaly_results`. Reasons are written first, models second.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.payroll.features import OT_HOURS_CAP, expected_gross
from src.payroll.detector import VERSION

OUTPUT_COLUMNS = ["result_id", "source_dataset", "record_id", "employee_id", "period", "anomaly_score", "flagged",
                  "reason_code", "reason_text", "money_at_risk", "model_version", "scored_at"]
SOURCE = "payroll_monthly_features"

# code -> sentence template. Codes that also exist in the sample sheet keep the sample's wording style.
REASONS = {
    "DUPLICATE_PAYMENT": "Second payslip paid to the same employee for the same month",
    "ZERO_PAY": "Active employee shows zero pay",
    "NEGATIVE_NET_PAY": "Deductions exceed gross pay",
    "GROSS_ABOVE_NORM": "Gross is {x:.1f}x what the fixed pay and days worked allow",
    "GROSS_BELOW_EXPECTED": "Gross is {x:.0f}% below what the fixed pay and days worked allow",
    "OVERTIME_ABOVE_CAP": "{x:g} paid overtime hours exceeds the 50-hour cap",
    "PF_NOT_DEDUCTED": "Provident fund missing although it was deducted in other months",
    "ROLE_PAY_ABOVE_PEERS": "Fixed pay is {x:.1f}x the usual pay for this role",
    "NET_PAY_MISMATCH": "Net pay does not equal gross pay minus deductions",
    "STATISTICAL_OUTLIER": "Unusual combination of pay ratios versus all other payslips",
}


def _fill(code: str, row, exp, ref) -> str:
    if code == "GROSS_ABOVE_NORM":
        return REASONS[code].format(x=row["gross_pay"] / exp)
    if code == "GROSS_BELOW_EXPECTED":
        return REASONS[code].format(x=(1 - row["gross_pay"] / exp) * 100)
    if code == "OVERTIME_ABOVE_CAP":
        return REASONS[code].format(x=row["overtime_hours_paid"])
    if code == "ROLE_PAY_ABOVE_PEERS":
        return REASONS[code].format(x=row["monthly_fixed_gross"] / ref["role_median_fixed"][row["role_id"]])
    return REASONS[code]


def money_at_risk(df: pd.DataFrame, codes: pd.Series, ref: dict) -> pd.Series:
    """Rupees that were (or would be) paid wrongly: the gap between expected and paid take-home pay."""
    base = expected_gross(df)
    role_med = df["role_id"].map(ref["role_median_fixed"])
    ot = df["overtime_hours_paid"].replace(0, np.nan)
    ot_trim = df["overtime_pay"] * (1 - OT_HOURS_CAP / ot).clip(lower=0).fillna(0)
    exp = base.where(codes != "ROLE_PAY_ABOVE_PEERS", expected_gross(df, fixed=role_med))
    exp = exp.where(codes != "OVERTIME_ABOVE_CAP", base - ot_trim)
    ratio = df["employee_id"].map(ref["emp_net_ratio"]).fillna(ref["global_net_ratio"])
    gap = (df["net_pay"] - exp * ratio).abs()
    own_pf = df["employee_id"].map(ref["emp_median_pf"]).fillna(0)
    out = gap.where(codes != "DUPLICATE_PAYMENT", df["net_pay"])
    out = out.where(codes != "PF_NOT_DEDUCTED", own_pf)
    out = out.where(codes != "NET_PAY_MISMATCH", (df["gross_pay"] - df["total_deductions"] - df["net_pay"]).abs())
    return out.round(2)


def build_results(df: pd.DataFrame, detector, scored: pd.DataFrame | None = None, scored_at: datetime | None = None,
                  only_flagged: bool = True) -> pd.DataFrame:
    """Flags in the official format, ranked by money at risk (biggest first).

    Tie-break when money is equal: higher anomaly score, then employee id, then record id (so output is repeatable).
    """
    scored = detector.score(df) if scored is None else scored
    ref = detector.ref_
    code = scored["rule_code"].where(scored["rule_flag"], "STATISTICAL_OUTLIER")
    d = df.copy()
    d["_exp"] = expected_gross(d)
    money = money_at_risk(d, code, ref)
    texts = [_fill(c, r, e, ref) for c, (_, r), e in zip(code, d.iterrows(), d["_exp"])]
    when = (scored_at or datetime.now(timezone.utc)).replace(microsecond=0, tzinfo=None)
    res = pd.DataFrame({
        "source_dataset": SOURCE, "record_id": d["payroll_item_id"].to_numpy(),
        "employee_id": d["employee_id"].to_numpy(), "period": pd.to_datetime(d["month"]).to_numpy(),
        "anomaly_score": scored["anomaly_score"].to_numpy(), "flagged": np.where(scored["flagged"], "Y", "N"),
        "reason_code": np.where(scored["flagged"], code, None), "reason_text": np.where(scored["flagged"], texts, None),
        "money_at_risk": np.where(scored["flagged"], money, np.nan), "model_version": VERSION, "scored_at": when})
    if only_flagged:
        res = res[res["flagged"] == "Y"]
    res = res.sort_values(["money_at_risk", "anomaly_score", "employee_id", "record_id"],
                          ascending=[False, False, True, True], na_position="last").reset_index(drop=True)
    res.insert(0, "result_id", [f"AN-{i + 1:04d}" for i in range(len(res))])
    return res[OUTPUT_COLUMNS]
