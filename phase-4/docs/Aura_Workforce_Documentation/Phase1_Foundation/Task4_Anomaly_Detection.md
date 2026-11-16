# Task 4: Attendance Anomaly Detection

**Phase 1 (Foundation) | Status: Done | Code:** `src/anomaly/inject.py`, `src/anomaly/detect.py`, `src/features/attendance.py` | **Notebook:** `task4_anomaly_detection.ipynb` | **Tests:** `tests/test_anomaly.py` | **Report:** `Task4_Documentation.pdf`

## 1. Objective
Find attendance months that look unusual (long absences, impossible hours, sudden pattern changes) without having labels, and measure how well the detector works.

## 2. Why it matters
Odd attendance can mean a data error, a person in trouble, or misuse. HR needs a short, trustworthy list, not hundreds of alerts.

## 3. Input data
Per-employee monthly attendance features built from the workbook (`attendance_monthly_features`), plus 150 labels shipped with the dataset that are kept for a second, independent exam only.

## 4. What was done, step by step
1. Built per-employee monthly attendance features with the Task 2 style fit/transform approach.
2. **Injected 140 known anomalies** with a fixed seed (42): long absence, impossible hours, sudden pattern change. Every injected row is logged in `injected_rows_log.csv`. With no labels you cannot measure quality, so injected rows give ground truth.
3. Trained an **Isolation Forest** (random trees; unusual points are isolated in fewer splits, so a shorter path means a higher anomaly score).
4. **Swept thresholds**: for each, recorded the catch rate and the false-alarm rate. Chose the best F1 among thresholds that flag at most 5% of rows, and wrote down why.
5. Saved the model, threshold and metadata (`attendance-anomaly-1.0.0`).
6. Ran the second exam on the 150 provided labels, never seen in training.

## 5. Deliverables
Notebook, `Task4_Documentation.pdf`, saved model and `.meta.json`, `injected_rows_log.csv`, threshold tables and precision-recall charts.

## 6. Key results
| Measure | Result |
|---|---|
| Chosen setting | Flag the top 3.5% of months |
| Catch rate on injected anomalies | 85% |
| False-alarm rate | 1.9% |
| Precision | 47% (about 14 flags per calendar month) |
| Long absence / pattern change | 100% caught |
| Impossible hours | 53% by the model (a rule "hours above 12" catches 45 of 45 with 0 false alarms) |
| Provided labels (never trained on) | 104 of 150 caught (69%) |

The weak spot is late-arrival spikes.

## 7. How to run
Open `task4_anomaly_detection.ipynb`, select the `.venv` kernel, Run All. Then `pytest -q`.

## 8. Interview questions
1. **How does Isolation Forest work?** Random trees split on random features. Unusual points are isolated in fewer splits, so a shorter path means a higher anomaly score.
2. **Why inject anomalies?** To get ground truth for catch rate and false alarms when no labels exist.
3. **Why not flag 20%?** HR would stop trusting the list.
4. **What is the model's weakness?** Single-feature anomalies (impossible hours, late spikes) are diluted, so hard rules help.
5. **Why is the injection repeatable?** A fixed random seed, and a test checks that two runs are identical.

## 9. Limits
Injected anomalies are obvious, so the 85% catch rate looks better than real life. The provided-label score (69%) is the more honest number.
