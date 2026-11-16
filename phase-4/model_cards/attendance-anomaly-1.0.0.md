# Model card: Attendance anomaly detector (`attendance-anomaly-1.0.0`)

| Field | Value |
|---|---|
| Version | attendance-anomaly-1.0.0 |
| Owner | Katarukonda Ashok (GM00154), AI/ML Engineer, Aura Workforce (CortexHR) |
| Status | Released as v1.0.0 for review; a support signal for people, never a decision |

## Purpose
Finds attendance months that look unusual (long absence, impossible hours, sudden change).

## Intended use and out-of-scope use
- **Use:** HR reviews flagged months to find data errors or people who may need support.
- **Do not use for:** Never for discipline. Long paid leave (for example maternity) is normal and must not be treated as a fault.

## Data
Per-employee monthly attendance features from the workbook; 140 injected anomalies for measurement; 150 provided labels as a second exam. All data is fictional (Zenith Nova Technologies).

## Method
Isolation Forest on per-employee features, flagging the top 3.5% of months, plus simple rules (for example hours above 12).

## Metrics
85% of injected anomalies caught at 1.9% false alarms (precision 47%); 69% (104 of 150) of the provided labels; weak on late-arrival spikes.

## Fairness
No protected attribute is used. Check that groups with legitimate patterns (shift workers, long leave) are not over-flagged.

## Limits (honest)
Injected anomalies are obvious, so 85% is optimistic; 69% is the honest number.

## Monitoring
PSI on attendance features.
