"""Create GROUND TRUTH by injecting known anomalies into attendance rows (repeatable with a fixed seed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
TYPES = ("LONG_ABSENCE", "IMPOSSIBLE_HOURS", "PATTERN_CHANGE")


def inject_anomalies(df: pd.DataFrame, n_per_type: dict | None = None, seed: int = SEED):
    """Return (table_with_injections, log).

    log has one row per injected row: id, employee, month, type, which columns changed and old/new values.
    Only rows with a healthy history (at least 15 scheduled days) are chosen so the injection is unambiguous.
    """
    n_per_type = n_per_type or {"LONG_ABSENCE": 50, "IMPOSSIBLE_HOURS": 45, "PATTERN_CHANGE": 45}
    rng = np.random.default_rng(seed)
    out = df.copy()
    eligible = out.index[(out["scheduled_days"] >= 15) & (out["attendance_rate"] >= 0.85)].to_numpy()
    total = sum(n_per_type.values())
    chosen = rng.choice(eligible, size=total, replace=False)
    log, cursor = [], 0
    for kind, n in n_per_type.items():
        for idx in chosen[cursor:cursor + n]:
            before = out.loc[idx].copy()
            sched = int(before["scheduled_days"])
            if kind == "LONG_ABSENCE":                 # many days absent in a row, unexplained
                days = int(rng.integers(7, 16))
                days = min(days, sched)
                out.loc[idx, "unpaid_absence_days"] = days
                out.loc[idx, "longest_unpaid_streak_days"] = days
                out.loc[idx, "present_days"] = max(0, int(before["present_days"]) - days)
                out.loc[idx, "attendance_rate"] = round(max(0.0, before["attendance_rate"] - days / sched), 4)
            elif kind == "IMPOSSIBLE_HOURS":           # more hours per day than a person can work
                out.loc[idx, "avg_hours_worked"] = round(float(rng.uniform(13.5, 21.0)), 2)
            elif kind == "PATTERN_CHANGE":             # sudden fall from this person's usual behaviour
                rate = float(rng.uniform(0.50, 0.75))
                out.loc[idx, "attendance_rate"] = round(rate, 4)
                out.loc[idx, "present_days"] = int(round(rate * sched))
                out.loc[idx, "unpaid_absence_days"] = int(rng.integers(2, 6))
                out.loc[idx, "longest_unpaid_streak_days"] = int(rng.integers(1, 3))
                out.loc[idx, "late_arrivals"] = int(before["late_arrivals"]) + int(rng.integers(6, 13))
                out.loc[idx, "avg_hours_worked"] = round(float(before["avg_hours_worked"]) - float(rng.uniform(1.0, 2.5)), 2)
            changed = [c for c in out.columns if before[c] != out.loc[idx, c]]
            log.append({"attendance_feature_id": before["attendance_feature_id"], "employee_id": before["employee_id"],
                        "month": before["month"], "injection_type": kind, "changed_columns": ";".join(changed),
                        "old_values": ";".join(str(before[c]) for c in changed),
                        "new_values": ";".join(str(out.loc[idx, c]) for c in changed)})
        cursor += n
    log = pd.DataFrame(log)
    out["is_injected"] = out["attendance_feature_id"].isin(log["attendance_feature_id"]).astype(int)
    return out, log
