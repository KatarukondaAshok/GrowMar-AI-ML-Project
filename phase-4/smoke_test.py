"""Task 17 smoke test: /v1/health plus ONE real prediction from every model, against a running API.

  python smoke_test.py                                  # tests http://127.0.0.1:8000
  python smoke_test.py https://your-api.onrender.com    # tests the live address (run it from your phone too)
If the API needs a key, set  AURA_API_KEY  first.  Exit code 0 = every check passed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AURA_API_URL", "http://127.0.0.1:8000")).rstrip("/")
KEY = os.environ.get("AURA_API_KEY")
FIX = Path(__file__).parent / "tests" / "fixtures"
HEAD = {"X-API-Key": KEY} if KEY else {}
RESUME = ("SUMMARY\nMachine learning engineer with 5 years of experience building models.\nSKILLS\nPython, SQL, Machine Learning, scikit-learn, Model Deployment\n"
          "EXPERIENCE\nML Engineer, Acme (5.0 yrs)\n")


def run() -> int:
    client = httpx.Client(base_url=BASE, headers=HEAD, timeout=90.0)    # long timeout: a sleeping free host needs about a minute to wake
    results = []

    def check(name, fn):
        t = time.time()
        try:
            ok, note = fn()
        except Exception as e:
            ok, note = False, f"{type(e).__name__}: {e}"
        results.append((name, ok, note, time.time() - t))

    def health():
        r = client.get("/v1/health"); j = r.json()
        return r.status_code == 200 and j["status"] == "ok" and j["models_loaded_times"] == 1, f"models: {len(j['models'])}, loaded {j['models_loaded_times']}x"

    def payroll():
        r = client.post("/v1/anomaly/payroll", json=json.loads((FIX / "payroll_request.json").read_text())); j = r.json()
        return r.status_code == 200 and j["records_flagged"] >= 1, f"flagged {j.get('records_flagged')} of {j.get('records_scored')}: {j['results'][0]['reason_code'] if j.get('results') else '-'}"

    def attendance():
        r = client.post("/v1/anomaly/attendance", json=json.loads((FIX / "attendance_request.json").read_text())); j = r.json()
        return r.status_code == 200 and j["records_flagged"] >= 1, f"flagged {j.get('records_flagged')}"

    def chat():
        r = client.post("/v1/chat", json={"session_id": "smoke", "message": "how many casual leaves do I have left"}, headers={**HEAD, "X-Demo-User": "EMP-0002"}); j = r.json()
        return r.status_code == 200 and j["intent"] == "leave_balance" and not j["refused"], f"intent={j.get('intent')} reply={j.get('reply')!r}"

    def chat_refusal():
        r = client.post("/v1/chat", json={"session_id": "smoke2", "message": "what is the salary of Priya Nair"}, headers={**HEAD, "X-Demo-User": "EMP-0002"}); j = r.json()
        return r.status_code == 200 and j["refused"] is True, f"refused={j.get('refused')}"

    def resume():
        r = client.post("/v1/resumes/rank", json={"job_id": "J04", "resumes": [{"resume_id": "A", "resume_text": RESUME}], "top": 1}); j = r.json()
        return r.status_code == 200 and len(j["ranked"]) == 1, f"score {j['ranked'][0]['score'] if j.get('ranked') else '-'}"

    def attrition():
        r = client.get("/v1/attrition/risk/EMP-0002"); j = r.json()
        return r.status_code == 200 and "risk_band" in j and "human review" in j["model_note"], f"{j.get('risk_band')} {j.get('risk_score_12m')}"

    def pay():
        r = client.post("/v1/pay/recommend", json={"role_id": "R005", "department_id": "D01", "job_level": 3, "band_min_2026": 900000, "band_mid_2026": 1100000,
                                                    "band_max_2026": 1400000, "prior_experience_years": 6, "skills_count": 6, "work_location": "Chennai",
                                                    "highest_education": "MCA"}); j = r.json()
        return r.status_code == 200 and j["low"] <= j["mid"] <= j["high"], f"{j.get('low')} to {j.get('high')}"

    def forecast():
        r = client.get("/v1/forecast", params={"metric": "headcount_end", "department_id": "ALL"}); j = r.json()
        return r.status_code == 200 and j["horizon_months"] == 6, f"6 months from {j['rows'][0]['forecast_month'][:7]}"

    def monitoring():
        r = client.get("/v1/monitoring"); j = r.json()
        return r.status_code == 200 and j["status"] in ("OK", "WATCH", "ALERT"), f"status {j.get('status')}"

    def bad_input():
        r = client.post("/v1/resumes/rank", json={"job_id": "bad", "resumes": []})
        return r.status_code == 422, f"HTTP {r.status_code}"

    for name, fn in [("health", health), ("anomaly payroll", payroll), ("anomaly attendance", attendance), ("chat answer", chat), ("chat refusal", chat_refusal),
                     ("resume rank", resume), ("attrition risk", attrition), ("pay bands", pay), ("forecast", forecast), ("monitoring", monitoring),
                     ("bad input gives 422", bad_input)]:
        check(name, fn)
    print(f"\nSMOKE TEST against {BASE}\n" + "-" * 78)
    for name, ok, note, secs in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<22} {secs:5.1f}s  {note}")
    failed = sum(1 for r in results if not r[1])
    print("-" * 78 + f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
