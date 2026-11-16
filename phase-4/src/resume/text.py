"""Task 11: read a resume without looking at who the person is.

A fair screener only needs skills, experience and job titles. Name, e-mail, city and college tier are removed
before any scoring (they are the only things that differ between the twin resumes in Task 12).
"""
from __future__ import annotations

import re

import pandas as pd

TIER = re.compile(r",?\s*Tier\s*\d\s*institution", re.I)
EXP_LINE = re.compile(r"\((\d+(?:\.\d+)?)\s*yrs?\)")
SUMMARY_YEARS = re.compile(r"with\s+(\d+(?:\.\d+)?)\s+years", re.I)


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def mask_resume(text: str, name: str | None = None, city: str | None = None) -> str:
    """Remove who the person is: name, e-mail, city and college tier, wherever they appear.

    Cutting the first lines is not enough: in two-column and table layouts the name sits on the same line as the
    first section (found by the Task 12 twin test), so the name and city given with the resume are removed by value.
    """
    t = str(text)
    parts = t.split("\n\n", 1)
    body = parts[1] if len(parts) == 2 else t
    body = EMAIL.sub(" ", body)
    for token in (name, city):
        if token and str(token) != "nan":
            body = re.sub(re.escape(str(token)), " ", body, flags=re.I)
    return re.sub(r"[ \t]{2,}", " ", TIER.sub("", body)).strip()


def split_vocab(cell) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return [s.strip() for s in re.split(r"[;,]", str(cell)) if s.strip()]


def build_vocabulary(jobs: pd.DataFrame, resumes: pd.DataFrame) -> list[str]:
    """Every skill named in a job description or listed by any candidate, longest first so 'MS Excel' wins over 'Excel'."""
    v = set()
    for col in ("required_skills", "nice_to_have_skills"):
        for cell in jobs[col]:
            v.update(split_vocab(cell))
    for cell in resumes["skills"]:
        v.update(split_vocab(cell))
    return sorted(v, key=lambda s: (-len(s), s.lower()))


def extract_skills(masked_text: str, vocabulary: list[str]) -> set[str]:
    """Skills mentioned in the (masked) text, matched on word boundaries, case-insensitive."""
    low = masked_text.lower()
    found = set()
    for skill in vocabulary:
        pat = r"(?<![a-z0-9+#])" + re.escape(skill.lower()) + r"(?![a-z0-9+#])"
        if re.search(pat, low):
            found.add(skill)
    return found


def extract_years(masked_text: str) -> float | None:
    """Total years: sum of '(2.2 yrs)' lines, else the summary sentence 'with 4.8 years'."""
    lines = [float(x) for x in EXP_LINE.findall(masked_text)]
    if lines:
        return round(sum(lines), 1)
    m = SUMMARY_YEARS.search(masked_text)
    return float(m.group(1)) if m else None
