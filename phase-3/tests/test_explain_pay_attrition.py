"""Tasks 13 to 15: reasons in plain English, pay-band output, saved attrition model."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attrition.explain import HIDDEN, reasons_for_row
from src.fairness import risk_bands

ROOT = Path(__file__).resolve().parents[1]
needs = pytest.mark.skipif(not (ROOT / "models/attrition-v2.joblib").exists(), reason="run: python scripts/run_phase3.py")


def test_reasons_put_risk_raising_drivers_first_and_use_real_numbers():
    g = pd.Series({"last_engagement_score": 0.4, "months_since_last_raise": 0.2, "tenure_months": 0.1, "comp_ratio": -0.3})
    x = pd.Series({"last_engagement_score": 2.2, "months_since_last_raise": 30, "tenure_months": 14, "comp_ratio": 1.1})
    r = reasons_for_row(g, x)
    assert r[0] == "Low engagement score (2.2/5)" and "30 months" in r[1] and "14 months" in r[2]
    assert all(len(s) > 10 for s in r)


def test_tenure_reason_describes_past_patterns_not_the_person():
    g = pd.Series({"tenure_months": 0.5}); r = reasons_for_row(g, pd.Series({"tenure_months": 8}))
    assert "in past data" in r[0]


def test_education_is_never_given_as_a_reason_and_missing_values_never_print_nan():
    g = pd.Series({"highest_education": 0.9, "last_engagement_score": 0.3, "comp_ratio": 0.1, "job_level": 0.05})
    x = pd.Series({"highest_education": 0, "last_engagement_score": np.nan, "comp_ratio": 0.8, "job_level": 3})
    r = reasons_for_row(g, x)
    assert "HIDDEN" and "education" not in " ".join(r).lower() and "nan" not in " ".join(r).lower() and HIDDEN == {"highest_education"}


def test_every_output_always_has_three_reasons_even_for_low_risk_people():
    g = pd.Series({"last_engagement_score": -0.4, "comp_ratio": -0.3, "tenure_months": -0.2, "job_level": -0.1})
    r = reasons_for_row(g, pd.Series({"last_engagement_score": 4.5, "comp_ratio": 1.1, "tenure_months": 90, "job_level": 3}))
    assert len(r) == 3 and all(r)


@needs
def test_saved_attrition_v2_gives_probabilities_and_the_feature_list_has_no_leakage():
    from src.attrition.v2 import AttritionModel
    m = AttritionModel.load(str(ROOT / "models/attrition-v2.joblib"))
    X = pd.read_csv(ROOT / "tests/fixtures/attrition_rows.csv")
    p = m.predict_proba(X)
    assert p.shape == (6,) and ((p > 0) & (p < 1)).all()
    assert (m.predict_proba(X) == p).all()                                    # deterministic
    assert not any(c in m.feature_names for c in ["exit_date", "exit_reason", "left_within_12m", "resignation_date", "fnf_settlement_amount"])
    assert not any("gender" in c or "age" in c.split("_") for c in m.feature_names)


@needs
def test_pay_band_is_ordered_inside_the_role_band_and_excludes_protected_features():
    from src.pay.model import FEATURES, PayBandModel
    m = PayBandModel.load(str(ROOT / "models/paybands-v1.joblib"))
    assert not any(k in FEATURES for k in ["gender", "career_gap_months", "age_years", "gender_female", "has_gap"])
    df = pd.DataFrame([{"role_id": "R001", "job_level": 2, "band_min_2026": 600000, "band_mid_2026": 750000, "band_max_2026": 975000, "prior_experience_years": 4.0, "skills_count": 5,
                        "market_index": 0.9, "hire_year": 2024, "department_id": "D01", "work_location": "Pune", "highest_education": "MBA"}])
    r = m.recommend(df).iloc[0]
    assert r.low <= r.mid <= r.high and 600000 <= r.low and r.high <= 975000
