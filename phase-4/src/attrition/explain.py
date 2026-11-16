"""Task 14: turn SHAP drivers into sentences an HR partner can read, with sensitive-data care."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

# feature -> (sentence when it RAISES risk, sentence when it LOWERS risk). {v} is the person's value.
TEXT = {
    "last_engagement_score": ("Low engagement score ({v:.1f}/5)", "Healthy engagement score ({v:.1f}/5)"),
    "engagement_missing": ("No engagement survey answer on record", "Engagement survey answer is on record"),
    "tenure_months": ("Shorter time at the company ({v:.0f} months): people with less time here left more often in past data",
                      "Long time at the company ({v:.0f} months): people with more time here left less often in past data"),
    "tenure_years": ("Shorter time at the company ({v:.1f} years): people with less time here left more often in past data",
                     "Long time at the company ({v:.1f} years): people with more time here left less often in past data"),
    "total_experience_years": ("Total experience ({v:.1f} years) matches groups that left more often", "Total experience ({v:.1f} years) matches groups that stayed"),
    "previous_employers_count": ("Number of earlier employers ({v:.0f}) matches groups that left more often", "Few earlier employers ({v:.0f})"),
    "commute_distance_km": ("Long commute ({v:.0f} km)", "Short commute ({v:.0f} km)"),
    "comp_ratio": ("Pay is low against the role band (ratio {v:.2f})", "Pay is healthy against the role band (ratio {v:.2f})"),
    "log_pay": ("Pay level matches groups that left more often", "Pay level matches groups that stayed"),
    "pay_vs_dept_median": ("Pay is below the department median", "Pay is above the department median"),
    "raises_received": ("Few pay raises so far ({v:.0f})", "Several pay raises so far ({v:.0f})"),
    "last_raise_pct": ("Small last raise ({v:.1f}%)", "Good last raise ({v:.1f}%)"),
    "months_since_last_raise": ("Long time since last raise ({v:.0f} months)", "Recent raise ({v:.0f} months ago)"),
    "no_raise_yet": ("No raise received yet", "Has received a raise"),
    "promotions_received": ("Few promotions so far ({v:.0f})", "Promoted {v:.0f} time(s)"),
    "months_since_last_promotion": ("Long time since last promotion ({v:.0f} months)", "Recent promotion ({v:.0f} months ago)"),
    "never_promoted": ("Never promoted", "Has been promoted"),
    "internal_transfers": ("Internal moves ({v:.0f}) match groups that left more often", "Internal moves ({v:.0f})"),
    "last_performance_rating": ("Last performance rating ({v:.1f}/5) matches groups that left more often", "Strong last performance rating ({v:.1f}/5)"),
    "not_reviewed_yet": ("No performance review on record", "Has a performance review on record"),
    "manager_changes_12m": ("Several manager changes in 12 months ({v:.0f})", "Stable manager ({v:.0f} changes in 12 months)"),
    "training_hours_12m": ("Little training in 12 months ({v:.0f} hours)", "Good training time in 12 months ({v:.0f} hours)"),
    "attendance_rate_3m": ("Lower attendance in the last 3 months ({v:.0%})", "Steady attendance in the last 3 months ({v:.0%})"),
    "attendance_gap": ("Attendance below the usual level", "Attendance at or above the usual level"),
    "late_arrivals_3m": ("More late arrivals in 3 months ({v:.0f})", "Few late arrivals in 3 months ({v:.0f})"),
    "avg_overtime_hours_per_month_3m": ("High overtime ({v:.0f} hours a month)", "Low overtime ({v:.0f} hours a month)"),
    "overtime_per_year_of_tenure": ("Overtime is high for the time spent here", "Overtime is moderate for the time spent here"),
    "avg_hours_worked_3m": ("Long working days ({v:.1f} hours)", "Normal working days ({v:.1f} hours)"),
    "wfh_share_3m": ("Share of work-from-home days ({v:.0%}) matches groups that left more often", "Share of work-from-home days ({v:.0%}) matches groups that stayed"),
    "unpaid_leave_days_6m": ("Unpaid leave in 6 months ({v:.0f} days)", "Little unpaid leave in 6 months ({v:.0f} days)"),
    "paid_leave_days_6m": ("Paid leave in 6 months ({v:.0f} days) matches groups that left more often", "Paid leave in 6 months ({v:.0f} days)"),
    "dept_attrition_rate_prior_12m": ("Department had more leavers in the last 12 months ({v:.0%})", "Department had few leavers in the last 12 months ({v:.0%})"),
    "job_level": ("Job level ({v:.0f}) matches groups that left more often", "Job level ({v:.0f}) matches groups that stayed"),
}
GROUP_TEXT = {  # one-hot groups: shown as one reason, without naming a personal characteristic
    "department_id": ("The team context has seen more leavers in past data", "The team context has seen fewer leavers in past data"),
    "work_location": ("The work location has seen more leavers in past data", "The work location has seen fewer leavers in past data"),
    "work_mode": ("The work arrangement has seen more leavers in past data", "The work arrangement has seen fewer leavers in past data"),
    "employment_type": ("The employment type has seen more leavers in past data", "The employment type has seen fewer leavers in past data"),
}
# Never shown as a reason: it describes the person's background, not something HR can or should act on.
HIDDEN = {"highest_education"}
GROUPS = ("department_id", "work_location", "work_mode", "employment_type", "highest_education")


def group_of(col: str) -> str:
    for g in GROUPS:
        if col.startswith(g + "_"):
            return g
    return col


def shap_values(booster, Xt: pd.DataFrame) -> np.ndarray:
    sv = shap.TreeExplainer(booster).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def grouped_shap(sv: np.ndarray, cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(sv, columns=cols)
    return df.T.groupby(group_of).sum().T


import re

_VALUE = re.compile(r"\s*\(\{v[^}]*\}[^)]*\)")


def _fmt(template: str, v) -> str:
    """Fill in the real value; if it is missing, leave the number out instead of printing nan."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return _VALUE.sub("", template).replace(": people", ": people")
    return template.format(v=v)


def reasons_for_row(g_row: pd.Series, x_row: pd.Series, k: int = 3) -> list[str]:
    """Top-k drivers of this person's score, risk-raising first. `x_row` holds the person's REAL values (raw data, plus
    tenure_years computed from months). Each reason is one plain sentence."""
    order = [c for c in g_row.sort_values(ascending=False).index if c not in HIDDEN]
    raising = [c for c in order if g_row[c] > 0][:k]
    chosen = raising + [c for c in reversed(order) if c not in raising][: max(0, k - len(raising))]   # fill with strongest protectors
    out = []
    for c in chosen:
        up = g_row[c] > 0
        if c in GROUP_TEXT:
            out.append(GROUP_TEXT[c][0 if up else 1] + ("" if up else ""))
        elif c in TEXT:
            out.append(_fmt(TEXT[c][0 if up else 1], x_row.get(c)))
        else:
            out.append(c.replace("_", " ").capitalize())
    return out + [""] * (k - len(out))
