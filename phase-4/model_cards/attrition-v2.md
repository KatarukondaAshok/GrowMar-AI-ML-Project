# Model card: Attrition risk model v2 (`attrition-v2`)

| Field | Value |
|---|---|
| Version | attrition-v2 |
| Owner | Katarukonda Ashok (GM00154), AI/ML Engineer, Aura Workforce (CortexHR) |
| Status | Released as v1.0.0 for review; a support signal for people, never a decision |

## Purpose
Estimates the chance an employee leaves within 12 months, to help HR start conversations earlier.

## Intended use and out-of-scope use
- **Use:** HR reads the band and the 3 reasons for people in the High and Medium bands and decides on support.
- **Do not use for:** Never for firing, pay, promotion or hiring decisions, and never shown to managers as a verdict.

## Data
`attrition_dataset` 2,383 employees after de-duplication, 14.6% left; leakage columns removed (exit date, exit reason and similar). All data is fictional (Zenith Nova Technologies).

## Method
Phase 1 feature pipeline, LightGBM (100 trees, 4 leaves), Platt calibration, SHAP for reasons. Same folds as the baseline.

## Metrics
5-fold ROC-AUC 0.725 vs 0.703 for the baseline; wins 3 of 5 folds; gain smaller than the fold-to-fold spread, so NOT proven. Locked test ROC-AUC 0.698 vs 0.691. Brier 0.114; calibration error 0.020.

## Fairness
Gender, age and other protected attributes are not features. Education is a feature but never shown as a reason. **Known issue to fix:** some Task 14 reasons show total experience (for example 'Total experience (31.8 years) matches groups that left more often'), which can act as an age proxy; reword or remove before wider use.

## Limits (honest)
Moderate ranking power. The latest monitor run shows drift on 'months since last raise' and 'attendance rate' (PSI 2.78 and 1.47), so 2026 scores are less reliable until the cause is known.

## Monitoring
PSI on 8 drivers every week; ALERT triggers the 4-step plan.
