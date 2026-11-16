# Phase 2: Core Development (Tasks 6 to 10)

**Folder:** `phase-2/` | **Tag:** `phase-2` | **Status:** Done and pushed (commit `3b73a45`) | **Report:** `phase-2/reports/phase2/AIML_AshokKatarukonda_Phase2_Report.pdf`

## What Phase 2 achieves
By Day 10 the project **detects, explains, serves and talks**: it finds payroll mistakes, gives every flag a reason and the money at risk, serves the detectors through a validated API, and answers HR questions with memory and permissions.

| Task | What it delivers | Document |
|---|---|---|
| 6 Payroll Detector | Rules plus Isolation Forest, measured per error type | Task 6 document |
| 7 Reason Codes | Reason code, plain sentence and money at risk in the official format | Task 7 document |
| 8 Anomaly API | `POST /v1/anomaly/payroll` and `/attendance` with validation and tests | Task 8 document |
| 9 Intent Classifier | TF-IDF plus logistic regression with a confidence fallback | Task 9 document |
| 10 Chatbot Slots and Permissions | Slots, memory, mock HR service, role rules | Task 10 document |

## How a flagged payslip travels
Raw payslip row, then ratio features, then a rule fires (or the model flags it), then a reason code, sentence and rupee amount are written, then it is ranked by money at risk, then the API returns it in the official format and the chatbot can read it out.

## Headline numbers
- 98 of 100 injected payroll errors caught (rules alone: precision 1.00, recall 0.93).
- 223 flags in the official 12-column format; the top flag is about Rs 21 lakh.
- Chatbot test accuracy 81.8%; with the 0.5 threshold it answers 65.9% of messages at 93.1% accuracy.
- 12 sample dialogues pass; a request for someone else's salary is always refused.
- 76 tests pass in the final folder, without needing the dataset.

## How to run Phase 2
```bash
cd phase-2
pip install -r requirements-dev.txt
pytest -q
uvicorn mock_services.mock_server:app --port 9000    # terminal 1
uvicorn app.main:app --port 8000                     # terminal 2
curl -s localhost:8000/health
```
Demo users: EMP-0002 and EMP-0003 (employees), EMP-0011 (HR). To rebuild the evidence with the full workbook, place it in `data/raw/` and run `scripts/train_phase2_models.py`, `build_mock_seed.py` and `run_phase2_evidence.py`.

## Still to do
Task 7 friend test (`Task7_Reason_Codes/FRIEND_FEEDBACK.md`), and rewriting the "What I learned" and "Problems" sections of the report in your own words.

## Honest notes
Rules do the work in Task 6. The combined detector's precision is 0.44. HR-analytics chatbot intents answer "not available yet" until Phase 3 features are connected. The starter `api_skeleton` and `mock_services` were not supplied, so both were built from the workbooks.
