"""Task 9 and 10: text cleaning, intent model, confidence fallback, slot extraction."""
from datetime import date

import pytest

from src.chat.nlu import IntentClassifier, clean_text
from src.chat.resources import load_resources
from src.chat.slots import extract_slots, mentions_other_persons_pay
from tests.conftest import MODELS, needs_models

TODAY = date(2026, 9, 30)
VOCAB = {"departments": ["Customer Support", "Engineering", "Sales", "Operations & IT"], "roles": ["Data Analyst", "Software Engineer"],
         "known_names": ["Priya Nair"]}


def test_clean_text_lowercases_strips_fillers_and_spaces():
    assert clean_text("  Quick question:   How MANY leaves?? pls ") == "how many leaves"


@needs_models
def test_model_understands_clear_and_typo_messages():
    m = IntentClassifier.load(str(MODELS / "intent-tfidf-lr-1.0.0.joblib"))
    got = m.predict(["what is the status of my laeve request", "suggest a slaary band for Accountant"])[0]
    assert got == ["leave_request_status", "salary_band_query"]


@needs_models
def test_low_confidence_gives_no_intent_so_the_bot_falls_back():
    m = IntentClassifier.load(str(MODELS / "intent-tfidf-lr-1.0.0.joblib"))
    assert m.predict(["zxqv blorp wumble"])[0] == [None]


@pytest.mark.parametrize("text,month", [("payslip for March", "2026-03"), ("show my attendance for last month", "2026-08"),
                                        ("payslip for this month", "2026-09"), ("salary slip of Dec", "2025-12"),
                                        ("may I know my leave balance", None), ("payslip for May", "2026-05")])
def test_month_slot(text, month):
    assert extract_slots(text, VOCAB, TODAY).get("month") == month


def test_other_slots():
    s = extract_slots("and sick leave for Support only", VOCAB, TODAY)
    assert s["leave_type"] == "SL" and s["department"] == "Customer Support"
    assert extract_slots("salary band for Data Analyst", VOCAB, TODAY)["role"] == "Data Analyst"


def test_person_name_plus_pay_word_is_someone_elses_pay():
    s = extract_slots("what is Priya Nair salary", VOCAB, TODAY)
    assert mentions_other_persons_pay("what is Priya Nair salary", s)
    s2 = extract_slots("what is Rahul Verma earning", VOCAB, TODAY)
    assert s2["person_name"] == "Rahul Verma" and mentions_other_persons_pay("what is Rahul Verma earning", s2)
    assert not mentions_other_persons_pay("show my payslip", extract_slots("show my payslip", VOCAB, TODAY))
