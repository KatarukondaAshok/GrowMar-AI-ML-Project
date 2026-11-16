# Task 17: Release to production: one live CortexHR API
Code: `app/main.py`, `app/schemas*.py`, `smoke_test.py`, `Dockerfile`, `start.sh`, `render.yaml`. Hosting reasoning: `HOSTING_DECISION.md`. Local proof: `smoke_test_output_local.txt`.

## Endpoint list (all under `/v1`)
| Method and path | Model | Notes |
|---|---|---|
| `GET /v1/health` | all | Models loaded once, versions listed. Open (no key) so hosts can check it |
| `POST /v1/anomaly/payroll` | payroll-anomaly-1.0.0 | 1 to 500 payslips, official output format |
| `POST /v1/anomaly/attendance` | attendance-anomaly-1.0.0 | 1 to 500 months |
| `POST /v1/chat` (header `X-Demo-User`) | intent-tfidf-lr-1.0.0 | Slots, memory, permissions, refuses others' salary |
| `POST /v1/resumes/rank` | resume-screener-1.0.0 | Masked ranking, no identity used |
| `GET /v1/attrition/risk/{employee_id}`, `GET /v1/attrition/summary` | attrition-v2 | 430 scored employees, band and 3 reasons |
| `POST /v1/attrition/score` | attrition-v2 | Scores raw employee rows (needs the 33 HR columns) |
| `POST /v1/pay/recommend` | paybands-v1 | Low, mid, high range |
| `GET /v1/forecast?metric=&department_id=` | headcount-forecast-1.0.0 | 6 months with 80% range |
| `GET /v1/monitoring` | monitor | Latest drift and fairness status |

## How it is built
- One FastAPI app; **all models load once** in the start-up hook (`/v1/health` shows `models_loaded_times: 1`).
- Pydantic validates input (bad input gives 422); unknown ids give 404.
- **Settings only from environment variables:** `AURA_MODELS_DIR`, `AURA_DATA_DIR`, `HR_SERVICE_URL`, `AURA_API_KEY`, `AURA_DEMO_TODAY`. No secret is in the repository (a test checks this). If `AURA_API_KEY` is set, every `/v1` call except health needs the header `X-API-Key`.
- `requirements.txt` pins every library version. `start.sh` runs the mock HR system (port 9000, internal) and the public API in one container.

## Smoke test
`python smoke_test.py https://YOUR-LIVE-URL` calls health and one real prediction for each model (11 checks). **Proof so far: 11/11 passed against the real running servers on this machine** (`smoke_test_output_local.txt`), and without the key the protected calls correctly returned 401. Memory measured: API about 198 MB + mock HR about 90 MB = about 290 MB.

## Still to do (only you can do these)
1. Deploy (steps in `HOSTING_DECISION.md`), set `AURA_API_KEY` in the host's dashboard.
2. Run `AURA_API_KEY=... python smoke_test.py https://YOUR-LIVE-URL` and also open `/v1/health` on your phone. Screenshot both.
3. Write the live link in the README and the report.

## Think for yourself: what breaks first with 100 callers?
My guess before testing: the single worker, because `/v1/anomaly/*` and `/v1/resumes/rank` do CPU work, and on a free host (0.1 CPU) calls queue up. Test it: `python scripts/load_test.py URL` fires 100 parallel calls and prints the slowest time and any failures.
**Local result (1 worker, a normal CPU):** 100 parallel resume-ranking calls, 100 ok, 0 failed, slowest 2.9 s (`load_test_local_1_worker.txt`). So on a normal CPU nothing breaks at 100 callers; calls simply queue behind each other. My guess for the free host (0.1 CPU) is that the slowest call grows to tens of seconds. **Run the same test on your live URL and write down what you saw**; that comparison is your evidence. Fixes if it is slow: `--workers 2`, a paid instance, or a rate limit.
