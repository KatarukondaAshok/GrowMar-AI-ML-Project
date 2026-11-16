"""Phase 4 request and response shapes (anomaly and chat shapes live in app/schemas.py from Phase 2)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

REVIEW_LABEL = "AI-generated, for human review"


class ResumeIn(BaseModel):
    resume_id: str = Field(min_length=1, max_length=40)
    resume_text: str = Field(min_length=20, max_length=20000)
    candidate_name: Optional[str] = Field(default=None, max_length=80)   # only used to REMOVE the name from the text
    city: Optional[str] = Field(default=None, max_length=60)


class RankRequest(BaseModel):
    job_id: str = Field(pattern=r"^J\d{2}$")
    resumes: list[ResumeIn] = Field(min_length=1, max_length=200)
    top: int = Field(default=5, ge=1, le=200)


class RankedResume(BaseModel):
    resume_id: str
    score: float
    required_coverage: float
    missing_required: list[str]
    years_found: Optional[float] = None


class RankResponse(BaseModel):
    job_id: str
    model_version: str
    ranked: list[RankedResume]
    note: str = "Support for a recruiter, not a decision. Name, gender, age, city and college tier are never used."
    review_label: str = REVIEW_LABEL


class AttritionRecords(BaseModel):
    model_config = ConfigDict(extra="allow")
    records: list[dict[str, Any]] = Field(min_length=1, max_length=500)


class PayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role_id: str = Field(pattern=r"^R\d{3}$")
    department_id: str = Field(pattern=r"^D\d{2}$")
    job_level: int = Field(ge=1, le=7)
    band_min_2026: float = Field(gt=0)
    band_mid_2026: float = Field(gt=0)
    band_max_2026: float = Field(gt=0)
    prior_experience_years: float = Field(ge=0, le=50)
    skills_count: int = Field(ge=0, le=60)
    work_location: str = Field(min_length=2, max_length=40)
    highest_education: Optional[str] = Field(default=None, max_length=80)


class PayResponse(BaseModel):
    model_version: str
    low: float
    mid: float
    high: float
    note: str = "Advice for a manager and HR, not a decision. A range, not one number."
    review_label: str = REVIEW_LABEL
