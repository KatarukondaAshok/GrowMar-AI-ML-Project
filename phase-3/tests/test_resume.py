"""Tasks 11 and 12: masking, extraction, scoring and the twin test (self-contained, no dataset needed)."""
import pandas as pd
import pytest

from src.resume.scoring import ResumeScreener
from src.resume.text import extract_skills, extract_years, mask_resume

JOBS = pd.DataFrame([{"job_id": "J1", "job_title": "Data Analyst", "required_skills": "Python; SQL; Excel", "nice_to_have_skills": "Tableau; Power BI",
                      "min_experience_years": 2, "description_text": "Data analyst role using Python and SQL."}])


def resume(rid, name, city, skills, tier="Tier 1", years=3.0, layout="clean"):
    exp = f"Data Analyst, Acme ({years} yrs)\n- Built reports"
    if layout == "two_column":      # name on the same line as the first section (as in the real two-column resumes)
        text = f"{name.upper()}      EXPERIENCE\n{city} | {name.lower().replace(' ', '.')}@mail.example   {exp}\nSKILLS\n{', '.join(skills)}\nEDUCATION\nB.Sc, {tier} institution"
    else:
        text = f"{name.upper()}\n{city} | {name.lower().replace(' ', '.')}@mail.example\n\nSUMMARY\nAnalyst with {years} years of experience.\n\nEXPERIENCE\n{exp}\n\nSKILLS\n{', '.join(skills)}\n\nEDUCATION\nB.Sc, {tier} institution"
    return {"resume_id": rid, "candidate_name": name, "city": city, "skills": "; ".join(skills), "resume_text": text}


def frame(*rows):
    return pd.DataFrame(rows)


@pytest.mark.parametrize("layout", ["clean", "two_column"])
def test_masking_removes_name_email_city_and_tier_in_every_layout(layout):
    r = resume("R1", "Pavithra Banerjee", "Hyderabad", ["Python"], layout=layout)
    m = mask_resume(r["resume_text"], r["candidate_name"], r["city"]).lower()
    assert "pavithra" not in m and "banerjee" not in m and "hyderabad" not in m and "@" not in m and "tier" not in m
    assert "python" in m and "data analyst" in m


def test_skill_and_year_extraction():
    r = resume("R1", "A B", "Pune", ["Python", "SQL", "Tableau"], years=3.5)
    m = mask_resume(r["resume_text"], "A B", "Pune")
    assert extract_skills(m, ["Python", "SQL", "Excel", "Tableau", "R"]) == {"Python", "SQL", "Tableau"}
    assert extract_years(m) == 3.5


def test_more_required_skills_means_a_higher_score():
    R = frame(resume("R1", "Asha Rao", "Pune", ["Python", "SQL", "Excel"]), resume("R2", "Ravi Nair", "Pune", ["Python"]), resume("R3", "Meena Iyer", "Pune", ["Cooking"]))
    top = ResumeScreener().fit(JOBS, R).rank(R, "J1", 3)
    assert top.resume_id.tolist() == ["R1", "R2", "R3"] and top.score.is_monotonic_decreasing


def test_twin_resumes_get_exactly_the_same_score_even_in_a_two_column_layout():
    R = frame(resume("O", "Pavithra Banerjee", "Pune", ["Python", "SQL"], tier="Tier 2", layout="two_column"),
              resume("T", "Aditya Krishnan", "Pune", ["Python", "SQL"], tier="Tier 1", layout="two_column"),
              resume("X", "Kavya Menon", "Pune", ["Excel"]))
    s = ResumeScreener().fit(JOBS, R).score_pairs(R, pd.DataFrame({"resume_id": ["O", "T"], "job_id": "J1"}))
    assert s.score.iloc[0] == s.score.iloc[1]


def test_screener_never_reads_columns_other_than_the_resume_body():
    R = frame(resume("A", "Pavithra Banerjee", "Pune", ["Python", "SQL"], tier="Tier 3"), resume("B", "Pavithra Banerjee", "Pune", ["Python", "SQL"], tier="Tier 1"))
    s = ResumeScreener().fit(JOBS, R).score_pairs(R, pd.DataFrame({"resume_id": ["A", "B"], "job_id": "J1"}))
    assert s.score.nunique() == 1          # college tier changes nothing
