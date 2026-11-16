# Phase 4: Release (Tasks 16 to 20)
Student: Katarukonda Ashok (GM00154) | Track: AI/ML Engineer (Aura Workforce, CortexHR) | Git tag: `phase-4`

By Day 20 the project **forecasts, goes live, watches itself, and has a face**: a 6-month forecast with ranges, ONE API for every model, a weekly drift and fairness monitor, a console for non-technical people, and a model card for every model.

## Results at a glance
| Task | Folder | Result |
|---|---|---|
| 16 Forecast | `Task16_Forecast` | 72-row forecast (company, 10 departments, attrition rate) with 80% ranges; headcount backtest error 2.33 vs 2.78 for the baseline |
| 17 Live API | `Task17_Live_API` | One `/v1` API with 11 endpoints for 7 models, loaded once; smoke test 11/11 on the real servers; Docker, Render config, pinned versions |
| 18 Monitoring | `Task18_Monitoring` | PSI on 8 features + twin-resume and pay-gap checks, OK/WATCH/ALERT, weekly schedule, `GET /v1/monitoring`. **Current status: ALERT (drift in 2 features), explained in its README** |
| 19 Console | `Task19_Console` | Gradio console, 7 tabs, review label on every output, friendly errors |
| 20 Handover | `Task20_Model_Cards_Handover` | 7 model cards, architecture, release notes, demo plan |

Report (one PDF): `reports/phase4/AIML_AshokKatarukonda_Phase4_Report.pdf`. Release notes: `RELEASE_NOTES.md`. Project documentation for every task of every phase: `docs/`.

## Run it (no dataset needed)
```bash
cd phase-4
pip install -r requirements-dev.txt
pytest -q                                  # 37 tests
PORT=8000 ./start.sh                       # mock HR system + the API   (Windows: use Docker, or run the two uvicorn lines inside start.sh in two terminals)
python smoke_test.py http://127.0.0.1:8000 # 11 checks
python -m src.monitor.monitor              # drift and fairness report
python console/app.py                      # the console on port 7860
```
Windows without Docker: terminal 1 `uvicorn mock_services.mock_server:app --port 9000`, terminal 2 `uvicorn app.main:app --port 8000`.
Settings are environment variables only (`AURA_API_KEY`, `AURA_MODELS_DIR`, `AURA_DATA_DIR`, `HR_SERVICE_URL`, `AURA_API_URL`). Nothing secret is committed.

## What is where
| Path | Content |
|---|---|
| `app/` | the one API (`main.py`) and its input and output shapes |
| `src/forecast/`, `src/monitor/` | Task 16 and 18 code |
| `src/payroll/`, `src/anomaly/`, `src/chat/`, `src/service.py` | Phase 2 code, copied so this folder runs alone |
| `src/resume/`, `src/attrition/`, `src/pay/`, `src/fairness.py`, `src/features/`, `src/models/` | Phase 1 and 3 code, copied |
| `models/` | 7 saved models (version in each name); `data/` serving tables and `data/monitor/` monitor extracts |
| `console/` | the Gradio console |
| `mock_services/` | mock HR system used by the chatbot |
| `model_cards/`, `docs/` | model cards and the documentation of all 14 tasks |
| `tests/` | 37 tests |

## Honest notes
- The unified package merges the Phase 2 and Phase 3 folders (same `src/` name); every saved model was loaded to check it still works.
- The saved models were built with scikit-learn 1.9.1; `requirements.txt` pins it.
- Deploying, scheduling the monitor, the stranger test, the demo video and the Final Challenge are **yours** (see each task README).
- The Docker image was written but not built here. The same code was tested by running `start.sh` and the smoke test against the real servers.
- The attrition reasons still include "total experience" (an age proxy); see `RELEASE_NOTES.md`.
