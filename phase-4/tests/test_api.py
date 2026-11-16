import json
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"
RESUME = "SUMMARY\nMachine learning engineer with 5 years of experience.\nSKILLS\nPython, SQL, Machine Learning\n"
PAY = {"role_id": "R005", "department_id": "D01", "job_level": 3, "band_min_2026": 900000, "band_mid_2026": 1100000, "band_max_2026": 1400000,
       "prior_experience_years": 6, "skills_count": 6, "work_location": "Chennai", "highest_education": "MCA"}


def test_health_loads_models_once(client):
    j = client.get("/v1/health").json()
    assert j["status"] == "ok" and j["models_loaded_times"] == 1 and len(j["models"]) >= 7
    client.get("/v1/health"); assert client.get("/v1/health").json()["models_loaded_times"] == 1


def test_payroll_attendance_official_results(client):
    p = client.post("/v1/anomaly/payroll", json=json.loads((FIX / "payroll_request.json").read_text())).json()
    assert p["records_flagged"] >= 1 and p["results"][0]["reason_code"] == "GROSS_ABOVE_NORM" and p["results"][0]["money_at_risk"] > 1e6
    a = client.post("/v1/anomaly/attendance", json=json.loads((FIX / "attendance_request.json").read_text())).json()
    assert a["records_flagged"] == 1


def test_chat_answer_and_refusal(client):
    ok = client.post("/v1/chat", json={"session_id": "t1", "message": "how many casual leaves do I have left"}, headers={"X-Demo-User": "EMP-0002"}).json()
    assert ok["intent"] == "leave_balance" and not ok["refused"]
    no = client.post("/v1/chat", json={"session_id": "t2", "message": "what is the salary of Priya Nair"}, headers={"X-Demo-User": "EMP-0002"}).json()
    assert no["refused"] is True


def test_resume_rank_orders_and_hides_identity(client):
    body = {"job_id": "J04", "top": 2, "resumes": [{"resume_id": "good", "resume_text": RESUME, "candidate_name": "Asha Rao", "city": "Pune"},
                                                    {"resume_id": "weak", "resume_text": "SUMMARY\nAccountant with 2 years of experience.\nSKILLS\nTally, GST\n"}]}
    j = client.post("/v1/resumes/rank", json=body).json()
    assert [r["resume_id"] for r in j["ranked"]] == ["good", "weak"] and "Asha" not in json.dumps(j) and "AI-generated" in j["review_label"]


def test_resume_twins_score_the_same(client):
    t = lambda n, c, tier: f"{n}\n{c} | x@y.example\nSUMMARY\nML engineer with 4 years of experience.\nSKILLS\nPython, SQL\nEDUCATION\nMCA, {tier} institution\n"
    body = {"job_id": "J04", "resumes": [{"resume_id": "A", "resume_text": t("ASHA RAO", "Pune", "Tier 1"), "candidate_name": "ASHA RAO", "city": "Pune"},
                                          {"resume_id": "B", "resume_text": t("RAVI KUMAR", "Chennai", "Tier 3"), "candidate_name": "RAVI KUMAR", "city": "Chennai"}]}
    s = {r["resume_id"]: r["score"] for r in client.post("/v1/resumes/rank", json=body).json()["ranked"]}
    assert s["A"] == pytest.approx(s["B"])


def test_attrition_risk_and_summary(client):
    r = client.get("/v1/attrition/risk/EMP-0004").json()
    assert r["risk_band"] in ("Low", "Medium", "High") and "human review" in r["model_note"]
    assert client.get("/v1/attrition/risk/EMP-9999").status_code == 404
    assert client.get("/v1/attrition/summary").json()["bands"] == {"Low": 301, "Medium": 86, "High": 43}


def test_pay_range_ordered_and_inside_band(client):
    j = client.post("/v1/pay/recommend", json=PAY).json()
    assert j["low"] <= j["mid"] <= j["high"] and 900000 <= j["low"] and j["high"] <= 1400000


def test_forecast_and_unknown_series(client):
    j = client.get("/v1/forecast", params={"metric": "headcount_end", "department_id": "D03"}).json()
    assert j["horizon_months"] == 6
    assert client.get("/v1/forecast", params={"metric": "nope"}).status_code == 404


def test_monitoring_endpoint(client):
    assert client.get("/v1/monitoring").json()["status"] in ("OK", "WATCH", "ALERT")


@pytest.mark.parametrize("path,body", [("/v1/resumes/rank", {"job_id": "bad", "resumes": []}), ("/v1/pay/recommend", {"role_id": "x"}),
                                       ("/v1/anomaly/payroll", {"records": []}), ("/v1/chat", {"session_id": "s"})])
def test_bad_input_is_422(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_unknown_job_404(client):
    assert client.post("/v1/resumes/rank", json={"job_id": "J99", "resumes": [{"resume_id": "a", "resume_text": RESUME}]}).status_code == 404


def test_api_key_required_when_set(keyed_client):
    assert keyed_client.get("/v1/health").status_code == 200                       # health stays open for the host's checks
    assert keyed_client.get("/v1/attrition/summary").status_code == 401
    assert keyed_client.get("/v1/attrition/summary", headers={"X-API-Key": "wrong"}).status_code == 401
    assert keyed_client.get("/v1/attrition/summary", headers={"X-API-Key": "secret-test-key"}).status_code == 200


def test_no_secret_files_in_repo():
    root = Path(__file__).resolve().parents[1]
    bad = [p for p in root.rglob(".env*") if ".venv" not in p.parts] + list(root.rglob("*.pem"))
    assert not bad
    assert "AURA_API_KEY" in (root / "render.yaml").read_text() and "sync: false" in (root / "render.yaml").read_text()
