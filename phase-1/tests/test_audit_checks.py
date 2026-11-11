import numpy as np
import pandas as pd
import pytest

from src.audit import checks as ck


@pytest.fixture
def tiny():
    return pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4, 5],
            "employee_id": ["A", "B", "C", "A", "D"],
            "role_id": ["R1"] * 5,
            "job_level": [1, 1, 1, 1, 1],
            "annual_fixed_pay": [100, 110, 1050, 100, 105],
            "comp_ratio": [1.0, 1.1, 1.05, 1.0, 1.05],
            "tenure_months": [12, -3, 24, 12, 30],
            "total_experience_years": [2.0, 3.0, 1.0, 2.0, 5.0],
            "attendance_rate_3m": [0.9, 1.12, 0.95, 0.9, 0.9],
            "wfh_share_3m": [0.1, -0.05, 1.02, 0.1, 0.3],
            "avg_hours_worked_3m": [8, 8, 8, 8, 2],
            "promotions_received": [0, 1, 0, 0, 2],
            "months_since_last_promotion": [np.nan, 5, np.nan, np.nan, 7],
            "raises_received": [0, 1, 1, 0, 2],
            "last_raise_pct": [np.nan, 5, 6, np.nan, 7],
            "last_performance_rating": [np.nan, 3, 3, np.nan, 4],
            "left_within_12m": [0, 1, 0, 0, 1],
            "voluntary_left_within_12m": [0, 1, 0, 0, 0],
            "exit_date": [pd.NaT, pd.Timestamp("2025-01-01"), pd.NaT, pd.NaT, pd.Timestamp("2025-02-01")],
            "resignation_date": [pd.NaT, pd.Timestamp("2024-12-01"), pd.NaT, pd.NaT, pd.NaT],
            "exit_reason": [None, "Compensation", None, None, "Relocation"],
            "exit_interview_done": [None, "Y", None, None, "N"],
            "fnf_settlement_amount": [np.nan, 5.0, np.nan, np.nan, 7.0],
        }
    )


def test_duplicates_found_and_dropped(tiny):
    assert len(ck.find_exact_duplicates(tiny)) == 2
    out = ck.drop_exact_duplicates(tiny)
    assert len(out) == 4 and out.employee_id.is_unique


def test_missing_table_only_columns_with_gaps(tiny):
    mt = ck.missing_table(tiny)
    assert "months_since_last_promotion" in mt.index
    assert "role_id" not in mt.index


def test_structural_missing_is_fully_explained(tiny):
    sm = ck.structural_missing_checks(tiny.assign(tenure_months=[3, 3, 24, 3, 30]))
    row = sm.set_index("column").loc["months_since_last_promotion"]
    assert row.gaps_explained == row.total_gaps


def test_outlier_methods_return_counts(tiny):
    oc = ck.outlier_compare(tiny, ["annual_fixed_pay"])
    assert oc.loc[0, "iqr_outliers"] >= 1


def test_salary_typo_detected(tiny):
    bad = ck.salary_typos(tiny)
    assert list(bad.employee_id) == ["C"]


def test_impossible_values_counts(tiny):
    iv = ck.impossible_values(tiny).set_index("rule").rows_failing
    assert iv["tenure_months < 0"] == 1
    assert iv["attendance_rate_3m > 1"] == 1
    assert iv["wfh_share_3m < 0"] == 1
    assert iv["wfh_share_3m > 1"] == 1


def test_clean_location():
    s = pd.Series(["BENGALURU", "Bangalore", "Madras", "delhi ncr", "Delhi-NCR", "Pune"])
    assert list(ck.clean_location(s)) == ["Bengaluru", "Bengaluru", "Chennai", "Delhi NCR", "Delhi NCR", "Pune"]


def test_class_balance(tiny):
    cb = ck.class_balance(tiny)
    assert cb["left"] == 2 and cb["stayed"] == 3 and cb["left_pct"] == 40.0


def test_leakage_columns_are_blank_for_stayers(tiny):
    lk = ck.leakage_table(tiny).set_index("column")
    assert lk.loc["exit_date", "filled_for_stayers_pct"] == 0.0
    assert lk.loc["exit_date", "filled_for_leavers_pct"] == 100.0
