"""Task 8: request and response shapes. Bad input never reaches a model: FastAPI answers 422 with the reason."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

EMP = r"^EMP-\d{4}$"
MAX_BATCH = 500        # rows per call: keeps one call fast (a few ms per row) and the payload small


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")      # an unknown field is a typo, not something to ignore


class PayrollRecord(_Strict):
    payroll_item_id: str = Field(min_length=3, max_length=20)
    employee_id: str = Field(pattern=EMP)
    month: date
    role_id: str = Field(pattern=r"^R\d{3}$")
    job_level: int = Field(ge=1, le=7)
    monthly_fixed_gross: float = Field(gt=0)
    payable_days: float = Field(ge=0, le=31)
    lop_days: float = Field(ge=0, le=31)
    overtime_hours_paid: float = Field(ge=0, le=200)
    overtime_pay: float = Field(ge=0)
    variable_pay: float = Field(ge=0)
    gross_pay: float = Field(ge=0)
    pf_employee: float = Field(ge=0)
    esi_employee: float = Field(ge=0)
    professional_tax: float = Field(ge=0)
    tds: float = Field(ge=0)
    other_deductions: float = Field(ge=0)
    total_deductions: float = Field(ge=0)
    net_pay: float                                  # can legitimately be negative: that is one of the errors
    run_id: Optional[str] = None
    department_id: Optional[str] = None
    overtime_eligible: Optional[str] = None
    variable_pay_type: Optional[str] = None


class AttendanceRecord(_Strict):
    attendance_feature_id: str = Field(min_length=3, max_length=20)
    employee_id: str = Field(pattern=EMP)
    month: date
    scheduled_days: int = Field(ge=0, le=31)
    present_days: int = Field(ge=0, le=31)
    wfh_days: int = Field(ge=0, le=31)
    half_days_unpaid: float = Field(ge=0, le=31)
    paid_leave_days: float = Field(ge=0, le=31)
    unpaid_absence_days: float = Field(ge=0, le=31)
    longest_unpaid_streak_days: float = Field(ge=0, le=31)
    late_arrivals: int = Field(ge=0, le=31)
    avg_hours_worked: float = Field(ge=0, le=24)
    overtime_hours: float = Field(ge=0, le=300)
    attendance_rate: float = Field(ge=0, le=1)
    department_id: Optional[str] = None
    job_level: Optional[int] = Field(default=None, ge=1, le=7)

    @model_validator(mode="after")
    def present_not_above_scheduled(self):
        if self.present_days > self.scheduled_days:
            raise ValueError("present_days cannot be more than scheduled_days")
        return self


class PayrollRequest(_Strict):
    records: list[PayrollRecord] = Field(min_length=1, max_length=MAX_BATCH)


class AttendanceRequest(_Strict):
    records: list[AttendanceRecord] = Field(min_length=1, max_length=MAX_BATCH)


class AnomalyResult(BaseModel):
    """Exactly the columns of the official sample sheet (sample_anomaly_results)."""
    result_id: str
    source_dataset: str
    record_id: str
    employee_id: str
    period: date
    anomaly_score: float
    flagged: str
    reason_code: Optional[str]
    reason_text: Optional[str]
    money_at_risk: Optional[float]
    model_version: str
    scored_at: datetime


class AnomalyResponse(BaseModel):
    model_version: str
    records_scored: int
    records_flagged: int
    results: list[AnomalyResult]


class ChatRequest(_Strict):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    message: str = Field(min_length=1, max_length=500)


class ChatResponse(BaseModel):
    session_id: str
    intent: Optional[str]
    confidence: float
    reply: str
    slots: dict
    clarify: bool
    refused: bool
    source: str
    model_version: str
