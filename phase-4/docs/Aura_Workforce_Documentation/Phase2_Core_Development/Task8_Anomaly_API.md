# Task 8: The Anomaly API (FastAPI)

**Phase 2 (Core Development) | Status: Done | Code:** `app/main.py`, `app/schemas.py`, `src/service.py` | **Tests:** `tests/test_api.py` | **Folder:** `phase-2/Task8_Anomaly_API`

## 1. Objective
Save the detectors and serve them through a validated web service that other systems can call.

## 2. Why it matters
A model that lives only in a notebook helps nobody. An API lets other systems use it safely, and versioned models let any result be traced to the exact model that made it.

## 3. What was built
| Endpoint | Purpose |
|---|---|
| `POST /v1/anomaly/payroll` | Score payslips (1 to 500 rows per call) |
| `POST /v1/anomaly/attendance` | Score attendance months (1 to 500 rows) |
| `POST /v1/chat` | The HR chatbot (Tasks 9 and 10) |
| `GET /health` | Shows the service is up and models loaded once |

Models are saved in `models/` with versions: `payroll-anomaly-1.0.0`, `attendance-anomaly-1.0.0`. Responses use the official sample columns.

## 4. What was done, step by step
1. Saved both detectors and the reason-code logic with `joblib`.
2. Wrote Pydantic schemas so bad input is rejected before any model runs.
3. Loaded the models **once** at start-up (lifespan hook), not on every request.
4. Wrote tests, including bad input that must return 422.
5. Ran the service and saved real `curl` commands and responses in `outputs/curl_session.md`.

## 5. Results and checks
- Both endpoints return the official format.
- Bad input returns 422 with the reason (6 parametrised cases plus the batch limit).
- Models load once (`/health` shows `models_loaded_times: 1`).
- 11 API tests pass (6 were required). The whole Phase 2 suite: 110 passed in the original build, 76 in the final folder.

## 6. How to run
```bash
python scripts/train_phase2_models.py
uvicorn mock_services.mock_server:app --port 9000     # mock HR system
uvicorn app.main:app --port 8000                      # the AI service
curl -s localhost:8000/health
```

## 7. Think for yourself: batch limits
One call takes 1 to 500 rows. Why 500: the payload stays small, one call finishes in milliseconds, and one caller cannot hog the service. Next steps: a request-size cap in the gateway, a rate limit per API key, and async jobs for larger files.

## 8. Interview questions
1. **Why load models at start-up?** Loading takes far longer than scoring; per-request loading would make every call slow.
2. **Why Pydantic?** Bad data is rejected with a clear 422 before it can reach a model.
3. **Why version models?** Any result can be traced to the exact model that made it (the `model_version` column).

## 9. Limits
The saved models were built with scikit-learn 1.9.1. If your environment is older, retrain with `python scripts/train_phase2_models.py`. The Dockerfile was not built in the environment where the folder was prepared.
