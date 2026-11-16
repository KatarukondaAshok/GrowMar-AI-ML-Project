import json
from pathlib import Path

import pandas as pd

from console import logic

FIX = Path(__file__).parent / "fixtures"


def test_every_output_carries_review_label(client):
    outs = [logic.risk_for(client, "EMP-0004"), logic.risks_summary(client)[1], logic.anomalies(client, "payroll", (FIX / "payroll_request.json").read_text())[1],
            logic.ask(client, "c1", "EMP-0002", "leave balance", [])[-1]["content"], logic.forecast_chart(client, "headcount_end", "ALL")[1],
            logic.monitoring(client)[1], logic.rank_resumes(client, "J04", "SUMMARY\nML engineer with 5 years of experience.\nSKILLS\nPython\n")[1]]
    assert all(logic.LABEL in o for o in outs)


def test_errors_are_friendly_not_crashes(client):
    assert "Nothing found" in logic.risk_for(client, "EMP-9999")
    assert "not valid JSON" in logic.anomalies(client, "payroll", "{broken")[1]
    assert "Paste at least one resume" in logic.rank_resumes(client, "J04", "  ")[1]
    assert "expected format" in logic.rank_resumes(client, "BAD", "SUMMARY\nML engineer.\nSKILLS\nPython here for sure\n")[1]
    assert "Type an employee ID" in logic.risk_for(client, "")


def test_unreachable_api_message():
    c = logic.make_client("http://127.0.0.1:1", timeout=1.0)
    assert "Could not reach" in logic.status(c)


def test_forecast_chart_and_rank_table(client):
    fig, note = logic.forecast_chart(client, "attrition_rate_monthly_pct", "ALL")
    assert fig is not None
    df, _ = logic.rank_resumes(client, "J04", "SUMMARY\nML engineer with 5 years of experience.\nSKILLS\nPython, SQL\n---\nSUMMARY\nAccountant.\nSKILLS\nTally, GST\n")
    assert list(df.resume_id) == ["Resume 1", "Resume 2"]


def test_console_app_file_is_valid_python():
    import ast
    ast.parse((Path(__file__).parents[1] / "console" / "app.py").read_text())
