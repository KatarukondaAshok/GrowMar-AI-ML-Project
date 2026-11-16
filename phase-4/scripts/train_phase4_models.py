"""Fit the resume screener (needs the dataset) and save it next to the other models. python scripts/train_phase4_models.py"""
from __future__ import annotations

from _common import ROOT, workbook

import joblib
import pandas as pd

from src.resume.scoring import ResumeScreener

x = pd.read_excel(workbook(), sheet_name=["job_descriptions", "resumes"])
sc = ResumeScreener(masked=True).fit(x["job_descriptions"], x["resumes"])
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump({"model": sc, "metadata": {"version": "resume-screener-1.0.0", "masked": True}}, ROOT / "models" / "resume-screener-1.0.0.joblib")
print("saved models/resume-screener-1.0.0.joblib")
