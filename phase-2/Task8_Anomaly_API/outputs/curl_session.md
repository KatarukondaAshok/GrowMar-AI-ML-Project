# Task 8: curl commands and real responses

Servers: mock HR system on :9000, AI service on :8000. Files: payroll_request.json, attendance_request.json (real rows from the workbook).

## 1. Health: models are loaded once
```
curl -s localhost:8000/health
{
    "status": "ok",
    "models": {
        "payroll": "payroll-anomaly-1.0.0",
        "attendance": "attendance-anomaly-1.0.0",
        "intent": "intent-tfidf-lr-1.0.0"
    },
    "models_loaded_times": 1
}
```

## 2. Payroll: PI-003304 (gross paid 10x) and PI-000001 (normal)
```
curl -s -X POST localhost:8000/v1/anomaly/payroll -H "Content-Type: application/json" -d @payroll_request.json
{
    "model_version": "payroll-anomaly-1.0.0",
    "records_scored": 2,
    "records_flagged": 1,
    "results": [
        {
            "result_id": "AN-0001",
            "source_dataset": "payroll_monthly_features",
            "record_id": "PI-003304",
            "employee_id": "EMP-0304",
            "period": "2025-12-01",
            "anomaly_score": 1.0,
            "flagged": "Y",
            "reason_code": "GROSS_ABOVE_NORM",
            "reason_text": "Gross is 10.0x what the fixed pay and days worked allow",
            "money_at_risk": 2102520.13,
            "model_version": "payroll-anomaly-1.0.0",
            "scored_at": "2026-11-12T02:47:54"
        }
    ]
}
```

## 3. Attendance: AF-03170 (no presence all month)
```
curl -s -X POST localhost:8000/v1/anomaly/attendance -H "Content-Type: application/json" -d @attendance_request.json
{
    "model_version": "attendance-anomaly-1.0.0",
    "records_scored": 1,
    "records_flagged": 1,
    "results": [
        {
            "result_id": "AN-0001",
            "source_dataset": "attendance_monthly_features",
            "record_id": "AF-03170",
            "employee_id": "EMP-0356",
            "period": "2025-07-01",
            "anomaly_score": 0.764,
            "flagged": "Y",
            "reason_code": "NO_ATTENDANCE",
            "reason_text": "No presence recorded for the whole month",
            "money_at_risk": null,
            "model_version": "attendance-anomaly-1.0.0",
            "scored_at": "2026-11-12T02:47:54"
        }
    ]
}
```

## 4. Bad input (payable_days = -3) gives HTTP 422 and the reason
```
curl -s -w "\nHTTP %{http_code}\n" -X POST localhost:8000/v1/anomaly/payroll -H "Content-Type: application/json" -d @bad_request.json
{
 "detail": [
  {
   "type": "greater_than_equal",
   "loc": [
    "body",
    "records",
    0,
    "payable_days"
   ],
   "msg": "Input should be greater than or equal to 0",
   "input": -3,
   "ctx": {
    "ge": 0.0
   }
  }
 ]
}
HTTP 422

```

## 5. Chat (Task 9): no sign-in, a policy question
```
{
    "session_id": "demo1",
    "intent": "wfh_policy",
    "confidence": 0.722,
    "reply": "Most roles follow a hybrid pattern with about three office days a week. Remote roles are listed in your offer letter.",
    "slots": {},
    "clarify": false,
    "refused": false,
    "source": "classifier",
    "model_version": "intent-tfidf-lr-1.0.0"
}
```
