"""Task 10: slot filling. Pull month, department, role, leave type and person name out of a message."""
from __future__ import annotations

import re
from datetime import date

from src.chat.resources import LEAVE_WORDS, MONTHS

PAY_WORDS = re.compile(r"\b(salary|pay|payslip|paysip|pasylip|ctc|earn|earns|earning|compensation|package|income|wage|wages|take[- ]home)\b", re.I)
NAME = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b")
NOT_NAMES = {"Casual", "Sick", "Earned", "Privilege", "Republic", "Customer", "Data", "Product", "Software"}


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def shift_month(d: date, k: int) -> date:
    i = d.year * 12 + (d.month - 1) + k
    return date(i // 12, i % 12 + 1, 1)


def department_aliases(departments: list[str]) -> list[tuple[str, str]]:
    """Full names first, then single words that belong to exactly one department ('support' -> Customer Support)."""
    full = [(d.lower(), d) for d in departments]
    words: dict[str, set] = {}
    for d in departments:
        for w in re.findall(r"[a-z]{4,}", d.lower()):
            words.setdefault(w, set()).add(d)
    short = [(w, next(iter(ds))) for w, ds in words.items() if len(ds) == 1 and w not in {"data"}]
    return sorted(full, key=lambda x: -len(x[0])) + sorted(short)


def extract_slots(text: str, vocab: dict, today: date) -> dict:
    """Return the slots found in `text`. `today` makes 'last month' repeatable in tests."""
    low = text.lower()
    slots: dict = {}
    this = month_start(today)
    if re.search(r"\blast month\b|\bprevious month\b", low):
        slots["month"] = shift_month(this, -1).strftime("%Y-%m")
    elif re.search(r"\bthis month\b|\bcurrent month\b", low):
        slots["month"] = this.strftime("%Y-%m")
    elif re.search(r"\bnext month\b", low):
        slots["month"] = shift_month(this, 1).strftime("%Y-%m")
    else:
        may_ok = re.search(r"\b(in|for|of|during)\s+may\b|\bmay\s+(payslip|salary|attendance|pay)\b", low)
        for w in re.findall(r"[a-z]+", low):
            if w in MONTHS and (w != "may" or may_ok):
                m = MONTHS[w]
                y = this.year if m <= this.month else this.year - 1     # the most recent such month
                slots["month"] = f"{y}-{m:02d}"
                break
    for alias, name in department_aliases(vocab.get("departments", [])):
        if re.search(rf"\b{re.escape(alias)}\b", low):
            slots["department"] = name; break
    for name in sorted(vocab.get("roles", []), key=len, reverse=True):
        if re.search(rf"\b{re.escape(name.lower())}\b", low):
            slots["role"] = name; break
    for phrase in sorted(LEAVE_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            slots["leave_type"] = LEAVE_WORDS[phrase]; break
    for n in vocab.get("known_names", []):
        if n.lower() in low:
            slots["person_name"] = n; break
    if "person_name" not in slots:
        known = {v.lower() for k in ("departments", "roles") for v in vocab.get(k, [])}
        for a, b in NAME.findall(text):
            if f"{a} {b}".lower() not in known and a not in NOT_NAMES and b not in NOT_NAMES:
                slots["person_name"] = f"{a} {b}"; break
    return slots


def mentions_other_persons_pay(text: str, slots: dict) -> bool:
    """Safety net: a named person plus a pay word means 'someone else's pay', whatever the classifier thinks."""
    return "person_name" in slots and bool(PAY_WORDS.search(text))


def month_label(ym: str) -> str:
    y, m = ym.split("-")
    return date(int(y), int(m), 1).strftime("%B %Y")
