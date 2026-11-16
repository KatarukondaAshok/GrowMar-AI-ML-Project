"""Write the small, ID-free extracts the monitor needs, so it can run on a schedule WITHOUT the programme workbook.
python scripts/export_monitor_extracts.py     (needs data/raw/02_CortexHR_AI_ML_Dataset.xlsx)"""
from __future__ import annotations

from _common import ROOT, workbook

import pandas as pd

from src.monitor.monitor import PSI_FEATURES

x = pd.read_excel(workbook(), sheet_name=None)
d = ROOT / "data" / "monitor"
d.mkdir(parents=True, exist_ok=True)
x["attrition_dataset"][PSI_FEATURES].to_csv(d / "psi_reference.csv", index=False)           # what the models were built on
x["attrition_scoring_2026"][PSI_FEATURES].to_csv(d / "psi_current.csv", index=False)         # what they score now
res, pairs = x["resumes"].set_index("resume_id"), x["resume_bias_pairs"]
ids = list(pairs.resume_id_original) + list(pairs.resume_id_twin)
res.loc[ids, ["candidate_name", "city", "skills", "resume_text"]].reset_index().to_csv(d / "twin_resumes.csv", index=False)
pairs[["pair_id", "resume_id_original", "resume_id_twin", "attribute_changed"]].to_csv(d / "twin_pairs.csv", index=False)
prot = x["protected_attributes"]; prot = prot[prot.record_type == "salary_hire"].set_index("person_ref")
s = x["salary_training"].copy(); s["gender"] = s["hire_id"].map(prot["gender"])
cols = ["department_id", "role_id", "job_level", "band_min_2026", "band_mid_2026", "band_max_2026", "prior_experience_years", "skills_count",
        "work_location", "highest_education", "market_index", "hire_year", "career_gap_months", "gender", "offered_annual_fixed"]
s[cols].to_csv(d / "pay_gap_inputs.csv", index=False)
jobs = x["job_descriptions"]; jobs.to_csv(d / "job_descriptions.csv", index=False)
print("written", sorted(p.name for p in d.iterdir()))
