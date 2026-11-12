"""Task 7: official columns, readable reasons, money at risk, ranking."""
from datetime import datetime

import pandas as pd

from src.payroll.reasons import OUTPUT_COLUMNS, REASONS, build_results
from tests.conftest import payslip

WHEN = datetime(2026, 10, 1, 2, 0)


def twenty_flagged(history, detector):
    rows = [payslip(7000 + i, emp=f"EMP-{(i % 12) + 1:04d}", month="2026-09-01", fixed=55000 + 1000 * (i % 12)) for i in range(20)]
    for k, r in enumerate(rows):                    # 20 different amounts of damage
        r["gross_pay"] = r["monthly_fixed_gross"] * (2 + k)
        r["net_pay"] = r["gross_pay"] - r["total_deductions"]
    return pd.DataFrame(rows)


def test_columns_match_the_official_sample_exactly(history, detector):
    r = build_results(twenty_flagged(history, detector), detector, scored_at=WHEN)
    assert list(r.columns) == ["result_id", "source_dataset", "record_id", "employee_id", "period", "anomaly_score", "flagged",
                               "reason_code", "reason_text", "money_at_risk", "model_version", "scored_at"] == OUTPUT_COLUMNS


def test_twenty_row_test_every_flag_has_a_reason_a_clerk_understands(history, detector):
    r = build_results(twenty_flagged(history, detector), detector, scored_at=WHEN).head(20)
    assert len(r) == 20
    assert r["reason_code"].isin(REASONS).all()
    assert r["reason_text"].str.len().between(15, 90).all()
    assert not r["reason_text"].str.contains("model said|score", case=False).any()    # reasons first, models second
    assert (r["money_at_risk"] > 0).all()


def test_ranked_by_money_at_risk_with_repeatable_ties(history, detector):
    df = twenty_flagged(history, detector)
    r = build_results(df, detector, scored_at=WHEN)
    assert r["money_at_risk"].is_monotonic_decreasing
    again = build_results(df.sample(frac=1, random_state=1), detector, scored_at=WHEN)
    assert r["record_id"].tolist() == again["record_id"].tolist()
    assert r["result_id"].iloc[0] == "AN-0001"


def test_duplicate_money_is_the_whole_second_payment(history, detector):
    dup = history.iloc[[0]].assign(payroll_item_id="PI-D00001")
    r = build_results(dup, detector, scored_at=WHEN)
    assert r.loc[0, "reason_code"] == "DUPLICATE_PAYMENT" and r.loc[0, "money_at_risk"] == dup["net_pay"].iloc[0]
