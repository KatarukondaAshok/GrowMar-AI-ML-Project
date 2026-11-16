# Task 13: Attrition Model v2: Beat the Baseline Honestly

**Phase 3 (Intelligence) | Status: Done | Code:** `src/attrition/v2.py` | **Saved model:** `models/attrition-v2.joblib` | **Folder:** `phase-3/Task13_Attrition_v2`

## 1. Objective
Build a stronger attrition model with gradient boosting and prove whether the gain is real.

## 2. Why it matters
Better predictions mean earlier help for employees, but only a gain without leakage is a real gain. A small honest gain beats a big fake one.

## 3. What was done, step by step
1. Reused the Phase 1 feature pipeline (leakage columns dropped, refitted inside every fold), the same 80/20 locked split and the **same 5 folds** as Task 3.
2. Trained LightGBM with 12 random settings, logged in `tuning_log.csv`, chosen by mean ROC-AUC on the folds (small trees: 100 trees, 4 leaves, learning rate 0.05).
3. Built one comparison table with the programme's `baseline_benchmarks` and the Task 3 baseline on the same folds.
4. Checked **calibration** (Platt scaling fitted on out-of-fold scores) and re-checked leakage.
5. Used the locked test set **once**.
6. Saved the model with a version string.

## 4. Comparison (5-fold CV on training rows)
| Model | ROC-AUC | PR-AUC | Precision (top 30%) | Recall (top 30%) |
|---|---|---|---|---|
| Reference: LogReg balanced (`baseline_benchmarks`) | 0.713 | 0.332 | n/a | n/a |
| My Task 3 baseline, same folds | 0.703 +/- 0.027 | 0.301 | 0.273 | 0.559 |
| **Attrition v2: LightGBM** | **0.725 +/- 0.030** | 0.339 | 0.284 | 0.581 |

"Top 30%" is the High plus Medium bands, the people HR would look at first.

## 5. Is the gain real?
Per-fold ROC-AUC difference (v2 minus baseline): -0.020, +0.069, -0.003, +0.047, +0.014. The mean is **+0.022**, the spread is 0.036, and v2 wins 3 of 5 folds. The mean gain is smaller than the fold-to-fold spread, so **a proven improvement cannot be claimed.** On the locked test set: v2 ROC-AUC 0.698 versus 0.691, but top-30% recall 0.514 versus 0.543. Mixed.

## 6. Calibration
Brier score 0.114 (always predicting the 14.6% base rate gives 0.125; the balanced logistic regression gives 0.207 because class weights inflate its probabilities). Calibration error 0.028 raw and **0.020 after Platt scaling**. A score of 0.20 can now be read as about 20 in 100.

## 7. Leakage re-check
No exit information is in the features (tested). The strongest single feature has ROC-AUC 0.61 (engagement, inverse), far from the 1.00 of the leaky model.

## 8. Think for yourself: ship a slightly better but harder-to-explain model?
Only because it is explainable: SHAP (Task 14) gives each prediction reasons, the probabilities are calibrated, and the gain is small. Ship v2 as a support signal, keep the baseline as a fallback, and say plainly that the ranking gain is within noise.

## 9. Interview questions
1. **How do you prove a model improvement is real?** Same folds, paired per-fold differences, compare the mean gain with the spread, then the locked test set once.
2. **Why calibrate?** So "0.20" means about 20 in 100, which HR can act on.
3. **How do you check for leakage?** Drop outcome columns in the pipeline, scan single-feature AUC, and be suspicious of any jump.
