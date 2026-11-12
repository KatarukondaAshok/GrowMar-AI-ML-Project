"""Task 8: both endpoints return the official format, bad input is a clear 422, models load once."""
import pytest

from tests.conftest import needs_models, payslip

pytestmark = needs_models
ATT = {"attendance_feature_id": "AF-X1", "employee_id": "EMP-0002", "month": "2026-08-01", "scheduled_days": 21,
       "present_days": 20, "wfh_days": 8, "half_days_unpaid": 0, "paid_leave_days": 1, "unpaid_absence_days": 0,
       "longest_unpaid_streak_days": 0, "late_arrivals": 1, "avg_hours_worked": 8.6, "overtime_hours": 0, "attendance_rate": 0.95}
COLS = {"result_id", "source_dataset", "record_id", "employee_id", "period", "anomaly_score", "flagged", "reason_code",
        "reason_text", "money_at_risk", "model_version", "scored_at"}


def api_payslip(i=1, **o):
    d = payslip(i, emp="EMP-0002", month="2026-09-01", role="R003", fixed=70000.0, **o)
    d.pop("run_id"); d.pop("department_id"); d.pop("overtime_eligible")
    d["month"] = "2026-09-01"
    return d


def test_payroll_flags_a_decimal_shift_in_the_official_format(api):
    body = {"records": [api_payslip(1), api_payslip(2, gross_pay=700000.0, net_pay=696000.0)]}
    r = api.post("/v1/anomaly/payroll", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["records_scored"] == 2
    top = j["results"][0]
    assert set(top) == COLS and top["record_id"] == "PI-000002" and top["flagged"] == "Y"
    assert top["reason_code"] == "GROSS_ABOVE_NORM" and top["money_at_risk"] > 100000
    assert top["model_version"] == "payroll-anomaly-1.0.0"


def test_attendance_flags_impossible_hours(api):
    r = api.post("/v1/anomaly/attendance", json={"records": [ATT, {**ATT, "attendance_feature_id": "AF-X2", "avg_hours_worked": 17.5}]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert [x["record_id"] for x in res] == ["AF-X2"] and res[0]["reason_code"] == "IMPOSSIBLE_HOURS"
    assert set(res[0]) == COLS and res[0]["money_at_risk"] is None


@pytest.mark.parametrize("path,bad", [
    ("/v1/anomaly/payroll", {"records": [{**api_payslip(), "payable_days": -3}]}),                   # negative days
    ("/v1/anomaly/payroll", {"records": [{**api_payslip(), "employee_id": "E-1"}]}),                  # wrong id shape
    ("/v1/anomaly/payroll", {"records": [{**api_payslip(), "surprise": 1}]}),                         # unknown field
    ("/v1/anomaly/payroll", {"records": []}),                                                        # nothing to score
    ("/v1/anomaly/attendance", {"records": [{**ATT, "present_days": 25}]}),                          # more present than scheduled
    ("/v1/anomaly/attendance", {"records": [{k: v for k, v in ATT.items() if k != "month"}]}),       # missing field
])
def test_bad_input_returns_422_with_a_reason(api, path, bad):
    r = api.post(path, json=bad)
    assert r.status_code == 422 and r.json()["detail"]


def test_batch_limit_is_enforced(api):
    r = api.post("/v1/anomaly/payroll", json={"records": [api_payslip(i) for i in range(501)]})
    assert r.status_code == 422


def test_models_load_once_not_per_request(api):
    for _ in range(3):
        api.post("/v1/anomaly/payroll", json={"records": [api_payslip()]})
    h = api.get("/health").json()
    assert h["models_loaded_times"] == 1 and h["models"]["payroll"] == "payroll-anomaly-1.0.0"


def test_chat_rejects_a_malformed_user_header_and_empty_message(api):
    assert api.post("/v1/chat", json={"session_id": "s", "message": "hi"}, headers={"X-Demo-User": "bob"}).status_code == 422
    assert api.post("/v1/chat", json={"session_id": "s", "message": ""}).status_code == 422
