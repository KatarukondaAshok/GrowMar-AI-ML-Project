# Task 8: The anomaly API (FastAPI)
Code: `app/main.py`, `app/schemas.py`, `src/service.py`. Models in `models/` (versioned: `payroll-anomaly-1.0.0`, `attendance-anomaly-1.0.0`).

## Run
```bash
python scripts/train_phase2_models.py            # once: saves models/
python scripts/build_mock_seed.py                # once: data for the mock HR system (needs data/raw workbooks)
uvicorn mock_services.mock_server:app --port 9000
uvicorn app.main:app --port 8000
```
Endpoints: `POST /v1/anomaly/payroll`, `POST /v1/anomaly/attendance` (body `{"records":[...]}`, 1 to 500 rows), `POST /v1/chat`, `GET /health`. Responses use the official sample columns. Real curl commands and responses: `outputs/curl_session.md`.

## Checklist
- Both endpoints return the official format (tests `test_api.py`).
- Bad input returns 422 with the reason (6 parametrised cases + batch limit).
- Models load once, in the lifespan hook (`/health` shows `models_loaded_times: 1`).
- 11 API tests pass (needed: 6). Whole suite: 110 passed.

## Think for yourself: batch limits
One call takes 1 to 500 rows (list is built in). Why 500: payload stays small, one call finishes in milliseconds, and one caller cannot hog the service. Next steps: request-size cap in the gateway, rate limit per API key, async jobs for larger files.

## Interview questions
1. *Why load models at start-up?* Loading takes far longer than scoring; per-request loading would make every call slow.
2. *Why Pydantic?* Bad data is rejected with a clear 422 before it can reach a model.
3. *Why version models?* Any result can be traced to the exact model that made it (`model_version` column).
