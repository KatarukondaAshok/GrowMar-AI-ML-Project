"""Task 8: reason codes for attendance flags, in the same official output format as payroll (Task 7)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.anomaly.detect import VERSION, anomaly_scores
from src.payroll.reasons import OUTPUT_COLUMNS

SOURCE = "attendance_monthly_features"
HOURS_LIMIT = 12          # nobody averages more than 12 hours a day (Task 4 finding)
LONG_STREAK_DAYS = 5
FREQUENT_LATE = 10
RATE_DROP = 0.15


def _reason(row, rate_vs_own: float) -> tuple[str, str]:
    if row["scheduled_days"] > 0 and row["present_days"] == 0:
        return "NO_ATTENDANCE", "No presence recorded for the whole month"
    if row["avg_hours_worked"] > HOURS_LIMIT:
        return "IMPOSSIBLE_HOURS", f"Average of {row['avg_hours_worked']:.1f} hours a day is not possible"
    if row["longest_unpaid_streak_days"] >= LONG_STREAK_DAYS:
        return "LONG_UNPAID_ABSENCE", f"{int(row['longest_unpaid_streak_days'])} consecutive unpaid absence days"
    if row["late_arrivals"] >= FREQUENT_LATE:
        return "FREQUENT_LATE", f"{int(row['late_arrivals'])} late arrivals this month"
    if rate_vs_own <= -RATE_DROP:
        return "PATTERN_SHIFT", "Attendance pattern differs sharply from own history"
    return "STATISTICAL_OUTLIER", "Unusual combination of attendance features versus all other months"


def build_attendance_results(df: pd.DataFrame, pipe, threshold: float, scored_at: datetime | None = None,
                             only_flagged: bool = True) -> pd.DataFrame:
    """Flag = Isolation Forest score >= threshold OR average hours above 12 (the rule Task 4 recommended)."""
    raw = anomaly_scores(pipe, df)
    feats = pipe[:-1].transform(df)
    rate_vs_own = pd.Series(feats[:, 12], index=df.index)   # column order = FEATURE_COLUMNS (rate_vs_own_mean)
    hours_rule = (df["avg_hours_worked"] > HOURS_LIMIT).to_numpy()
    model_flag = raw >= threshold
    flagged = model_flag | hours_rule
    rs = [_reason(r, rate_vs_own.loc[i]) for i, r in df.iterrows()]
    when = (scored_at or datetime.now(timezone.utc)).replace(microsecond=0, tzinfo=None)
    res = pd.DataFrame({
        "source_dataset": SOURCE, "record_id": df["attendance_feature_id"].to_numpy(),
        "employee_id": df["employee_id"].to_numpy(), "period": pd.to_datetime(df["month"]).to_numpy(),
        "anomaly_score": np.where(hours_rule, 1.0, np.round(raw, 3)), "flagged": np.where(flagged, "Y", "N"),
        "reason_code": np.where(flagged, [r[0] for r in rs], None),
        "reason_text": np.where(flagged, [r[1] for r in rs], None),
        "money_at_risk": np.nan, "model_version": VERSION, "scored_at": when})
    if only_flagged:
        res = res[res["flagged"] == "Y"]
    res = res.sort_values(["anomaly_score", "employee_id", "record_id"], ascending=[False, True, True]).reset_index(drop=True)
    res.insert(0, "result_id", [f"AN-{i + 1:04d}" for i in range(len(res))])
    return res[OUTPUT_COLUMNS]
