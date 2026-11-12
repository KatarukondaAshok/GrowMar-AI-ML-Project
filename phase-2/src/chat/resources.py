"""Chatbot reference data (reply templates, FAQ, vocabulary) saved next to the model so the API needs no Excel file."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MONTHS = {"jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
          "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
          "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12}
LEAVE_WORDS = {"casual leave": "CL", "casual": "CL", "cl": "CL", "sick leave": "SL", "sick": "SL", "sl": "SL",
               "earned leave": "EL", "privilege leave": "EL", "earned": "EL", "privilege": "EL", "el": "EL"}


def build_resources(cortex_xlsx: str, departments: list[str] | None = None, extra_roles: list[str] | None = None) -> dict:
    """Read the chatbot sheets once and keep only what the bot needs."""
    x = pd.read_excel(cortex_xlsx, sheet_name=["chatbot_responses", "hr_faq", "chatbot_nlu_training", "job_descriptions"])
    resp = x["chatbot_responses"].where(x["chatbot_responses"].notna(), None)
    responses = {r["intent"]: {k: r[k] for k in ["response_template", "requires_login", "allowed_roles",
                                                 "backend_or_service_call", "on_failure", "policy_note"]}
                 for _, r in resp.iterrows()}
    ents = {"department": set(), "role": set(), "person_name": set()}
    for e in x["chatbot_nlu_training"]["entities"].dropna():
        for d in json.loads(e):
            if d["entity"] in ents:
                ents[d["entity"]].add(d["normalized"])
    roles = sorted(ents["role"] | set(x["job_descriptions"]["job_title"]) | set(extra_roles or []))
    depts = sorted(ents["department"] | set(departments or []))
    faq = x["hr_faq"].to_dict("records")
    return {"responses": responses, "faq": faq, "departments": depts, "roles": roles,
            "known_names": sorted(ents["person_name"])}


def save_resources(res: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(res, indent=2, default=str))


def load_resources(path: str) -> dict:
    return json.loads(Path(path).read_text())
