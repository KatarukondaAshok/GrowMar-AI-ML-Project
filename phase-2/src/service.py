"""Task 8: the one place that turns request rows into official anomaly results (used by the API and the chatbot)."""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from src.anomaly.detect import VERSION as ATTENDANCE_VERSION, load_detector
from src.anomaly.reasons import build_attendance_results
from src.payroll.detector import VERSION as PAYROLL_VERSION, PayrollDetector
from src.payroll.reasons import build_results as build_payroll_results


def _jsonable(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        r["period"] = pd.Timestamp(r["period"]).date().isoformat()
        r["scored_at"] = pd.Timestamp(r["scored_at"]).to_pydatetime().isoformat()
        if isinstance(r.get("money_at_risk"), float) and math.isnan(r["money_at_risk"]):
            r["money_at_risk"] = None
        out.append(r)
    return out


class AnomalyService:
    """Holds both detectors. Built once at start-up, never per request."""

    def __init__(self, payroll: PayrollDetector, attendance_pipe, attendance_threshold: float):
        self.payroll_detector, self.att_pipe, self.att_threshold = payroll, attendance_pipe, attendance_threshold
        self.versions = {"payroll": PAYROLL_VERSION, "attendance": ATTENDANCE_VERSION}

    @classmethod
    def load(cls, models_dir: str) -> "AnomalyService":
        pay = PayrollDetector.load(f"{models_dir}/{PAYROLL_VERSION}.joblib")
        pipe, thr, _ = load_detector(f"{models_dir}/{ATTENDANCE_VERSION}.joblib")
        return cls(pay, pipe, thr)

    def payroll(self, records: list[dict], scored_at: datetime | None = None, only_flagged: bool = True) -> list[dict]:
        df = pd.DataFrame(records)
        df["month"] = pd.to_datetime(df["month"])
        res = build_payroll_results(df, self.payroll_detector, scored_at=scored_at, only_flagged=only_flagged)
        return _jsonable(res)

    def attendance(self, records: list[dict], scored_at: datetime | None = None, only_flagged: bool = True) -> list[dict]:
        df = pd.DataFrame(records).reset_index(drop=True)
        df["month"] = pd.to_datetime(df["month"])
        res = build_attendance_results(df, self.att_pipe, self.att_threshold, scored_at=scored_at, only_flagged=only_flagged)
        return _jsonable(res)
