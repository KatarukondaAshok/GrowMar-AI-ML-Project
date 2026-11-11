# Task 4 - Attendance Anomaly Detection

**Documentation:** `Task4_Documentation.pdf` | **Notebook:** `task4_anomaly_detection.ipynb` | **Code:** `src/anomaly/inject.py`, `src/anomaly/detect.py`, `src/features/attendance.py` | **Tests:** `tests/test_anomaly.py`

## What it does
Builds per-employee monthly attendance features, injects 140 known anomalies (long absence, impossible hours, sudden pattern change) with a fixed seed and logs them, trains an Isolation Forest, sweeps thresholds (catch rate vs false alarms), chooses one with a written reason, saves model + threshold + metadata, and runs a second exam on the 150 labels shipped with the dataset.

## Key results
- Chosen: flag the top 3.5% of months. Catch rate 85%, false-alarm rate 1.9%, precision 47%, about 14 flags per calendar month.
- By type: long absence 100%, pattern change 100%, impossible hours 53% (a simple rule "hours > 12" catches 45 of 45 with 0 false alarms).
- Provided labels (never seen in training): 104 of 150 caught (69%); weak spot is late-arrival spikes.
- Saved model: `outputs/artifacts/attendance-anomaly-1.0.0.joblib` + `.meta.json`. Injected rows: `outputs/tables/injected_rows_log.csv`.

## Likely review questions
1. **How does Isolation Forest work?** Random trees split on random features; unusual points are isolated in fewer splits, so a shorter path means a higher anomaly score.
2. **Why inject anomalies?** With no labels you cannot measure quality; injected rows give ground truth for catch rate and false alarms.
3. **Why not flag 20%?** HR would stop trusting the list. We pick best F1 among thresholds flagging at most 5% of rows.
4. **What is the model's weakness?** Single-feature anomalies (impossible hours, late spikes) are diluted; add hard rules.
5. **Why is the injection repeatable?** A fixed random seed (42); a test checks two runs are identical.
