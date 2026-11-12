# Phase 2: Core Development (Tasks 6 to 10)

Student: Katarukonda Ashok | Track: AI/ML Engineer (Aura Workforce, CortexHR) | Git tag: `phase-2`

By Day 10 the project **detects, explains, serves and talks**: it finds payroll mistakes, gives every flag a reason and the money at risk, serves the detectors through a validated API, and answers HR questions with memory and permissions.

## Results at a glance
| Task | Folder | Result |
|---|---|---|
| 6 Payroll outlier detector | [Task6_Payroll_Detector](Task6_Payroll_Detector) | Rules + Isolation Forest catch **98 of 100** injected errors (recall 0.98; baseline 0.40). Rules alone: precision 1.00 |
| 7 Reason codes and money at risk | [Task7_Reason_Codes](Task7_Reason_Codes) | 223 flags in the official 12-column format, ranked by money; top flag about Rs 21 lakh |
| 8 Anomaly API | [Task8_Anomaly_API](Task8_Anomaly_API) | `POST /v1/anomaly/payroll` and `/attendance`, 422 on bad input, models load once, 11 API tests |
| 9 HR intent classifier | [Task9_HR_Intent_Classifier](Task9_HR_Intent_Classifier) | 81.8% test accuracy; confidence threshold 0.5 answers 66% of messages at 93% accuracy, falls back politely on the rest |
| 10 Slots, memory, permissions | [Task10_Chatbot_Slots_Permissions](Task10_Chatbot_Slots_Permissions) | 12 sample dialogues pass; someone else's salary is always refused |

Report (one PDF): [reports/phase2/AIML_AshokKatarukonda_Phase2_Report.pdf](reports/phase2/AIML_AshokKatarukonda_Phase2_Report.pdf)

## How to run (from inside this folder)
```bash
cd phase-2
pip install -r requirements-dev.txt
pytest -q                                           # 76 tests, no dataset needed
uvicorn mock_services.mock_server:app --port 9000   # terminal 1: mock HR system (small sample data)
uvicorn app.main:app --port 8000                    # terminal 2: the AI service
curl -s localhost:8000/health
```
Try the API with the real request files:
```bash
curl -s -X POST localhost:8000/v1/anomaly/payroll -H "Content-Type: application/json" -d @Task8_Anomaly_API/outputs/payroll_request.json
curl -s -X POST localhost:8000/v1/chat -H "Content-Type: application/json" -H "X-Demo-User: EMP-0002" \
     -d '{"session_id":"demo","message":"how many casual leaves do I have left"}'
```
Demo users in the sample data: `EMP-0002` and `EMP-0003` (employees), `EMP-0011` (HR).
To retrain or rebuild the evidence with the full workbooks, put them in `data/raw/` (see `data/raw/README.md`) and run `python scripts/train_phase2_models.py`, `python scripts/build_mock_seed.py`, `python scripts/run_phase2_evidence.py`. The workbooks are not committed.

## What is where
| Path | Content |
|---|---|
| `src/payroll/` | features, rules, detector, reasons and money at risk, evaluation (Tasks 6, 7) |
| `app/`, `src/service.py` | FastAPI app, schemas, anomaly service (Task 8) |
| `src/chat/` | cleaning, intent model, FAQ, slots, sessions, HR client, bot (Tasks 9, 10) |
| `src/anomaly/`, `src/features/` | Phase 1 attendance detector, reused by the API (copied so this folder runs alone) |
| `mock_services/` | mock HR system used by the chatbot |
| `models/` | saved, versioned models: `payroll-anomaly-1.0.0`, `attendance-anomaly-1.0.0`, `intent-tfidf-lr-1.0.0` |
| `Dockerfile`, `pyproject.toml`, `requirements*.txt` | container, package and dependency files (same set as phase-1; the Dockerfile has not been built here) |
| `tests/` | 76 tests: rules, output format, API, intent model, slots, dialogues, permissions |
| `reports/phase2/` | Phase 2 report PDF and `results.json` |

## Honest notes
- The rules do most of the work in Task 6; the model adds 5 of 98 catches among 130 model-only flags. Precision of the combined detector is 0.44.
- Anomaly labels are synthetic, so real-world recall will be lower. No label is used to train any model.
- HR analytics intents (attrition, salary band, headcount, resume score) are recognised and role-checked but answer "not available yet" until Phase 3.
- The starter `api_skeleton` and `mock_services` were not supplied, so both were built from the workbooks.
- Task 7's friend test is a TODO: see `Task7_Reason_Codes/FRIEND_FEEDBACK.md`.
