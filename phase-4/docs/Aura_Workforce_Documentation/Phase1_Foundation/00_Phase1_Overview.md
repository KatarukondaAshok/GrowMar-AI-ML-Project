# Phase 1: Foundation (Tasks 1 to 4)

**Folder:** `phase-1/` | **Tag:** `phase-1` | **Status:** Done and pushed | **First review date:** 17 October 2026

## What Phase 1 achieves
Phase 1 makes the data safe to model and sets the benchmark every later model must beat. It answers four questions, one per task.

| Task | Question it answers | Document |
|---|---|---|
| 1 Data Audit | Is the data fit to model? | Task 1 document |
| 2 Feature Pipeline | Can every model prepare data the same way? | Task 2 document |
| 3 Baseline Models | What score must later models beat? | Task 3 document |
| 4 Anomaly Detection | Which attendance months look unusual? | Task 4 document |

## How the tasks build on each other
Task 1's 26 decisions become Task 2's code. Task 2's pipeline feeds Task 3's baselines. Task 4 reuses the same fit-and-transform style for attendance.

## Headline numbers
- 2,383 employees after removing 6 duplicates; 14.6% left; 6 leakage columns found.
- 74 model-ready columns, 0 missing values.
- Baseline: balanced logistic regression, ROC-AUC about 0.70, recall about 0.61.
- Isolation Forest flags 3.5% of attendance months and catches 85% of injected anomalies with 1.9% false alarms.

## How to run Phase 1
```bash
cd phase-1
python -m venv .venv && .venv\Scripts\activate
pip install --no-cache-dir -r requirements.txt
# open each taskN_*.ipynb, select the .venv kernel, Run All
pytest -q          # 34 tests
```
Docker files are provided but were not executed where the folder was prepared; run `docker build` once before a review.

## Deliverables
Four notebooks, four documentation PDFs (`Task1_Documentation.pdf` to `Task4_Documentation.pdf`), shared code in `src/`, tests in `tests/`, and the Phase 1 report PDF.

## Honest notes
The workbook contains answer-key sheets. Findings were produced from the data first and compared with those sheets only afterwards. The injected anomalies in Task 4 are synthetic and obvious, so the provided-label score (69%) is the more honest one.
