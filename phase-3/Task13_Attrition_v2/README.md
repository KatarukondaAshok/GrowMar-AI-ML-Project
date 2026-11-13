# Task 13: Attrition model v2 (gradient boosting), honestly compared
Code: `src/attrition/v2.py`. Evidence: `outputs/tables/`. Saved model: `models/attrition-v2.joblib`. **Result: a small, not clearly real gain over the Task 3 baseline. Calibrated probabilities are the clearer benefit.**

## Steps
1. Same Phase 1 feature pipeline (leakage columns dropped; refit inside every fold), same 80/20 locked split and the **same 5 folds** as Task 3 (stratified, shuffled, seed 42).
2. LightGBM with 12 random settings, logged in `tuning_log.csv`; chosen by mean ROC-AUC on the folds (small trees: 100 trees, 4 leaves, learning rate 0.05).
3. One comparison table with `baseline_benchmarks` and my own baseline on the same folds.
4. Calibration (Platt scaling fitted on out-of-fold scores) and a leakage re-check.
5. The locked test set is used **once**.

## Comparison (5-fold CV on training rows)
| Model | ROC-AUC | PR-AUC | Precision (top 30%) | Recall (top 30%) |
|---|---|---|---|---|
| Reference: LogReg balanced (`baseline_benchmarks`) | 0.713 | 0.332 | n/a | n/a |
| My Task 3 baseline: LogReg balanced, same folds | 0.703 +/- 0.027 | 0.301 | 0.273 | 0.559 |
| **Attrition v2: LightGBM** | **0.725 +/- 0.030** | 0.339 | 0.284 | 0.581 |

"Top 30%" = the High + Medium bands (the people HR would look at first).

## Is the gain real?
Per-fold ROC-AUC difference (v2 minus baseline): -0.020, +0.069, -0.003, +0.047, +0.014. Mean **+0.022**, spread 0.036, v2 wins 3 of 5 folds. The mean gain is smaller than the fold-to-fold spread, so **I cannot claim a proven improvement**. On the locked test set (used once): v2 ROC-AUC 0.698 vs 0.691, PR-AUC 0.358 vs 0.323, but top-30% recall 0.514 vs 0.543 and precision 0.252 vs 0.266. Mixed.

## Calibration
Brier score 0.114 (always predicting the 14.6% base rate gives 0.125; the balanced LogReg gives 0.207 because class weights inflate its probabilities). Expected calibration error 0.028 raw, **0.020 after Platt scaling** (`figures/reliability.png`). A score of 0.20 can now be read as about 20%.

## Leakage re-check
No exit information is in the features (tested). Strongest single feature has ROC-AUC 0.61 (engagement, inverse), far from the 1.00 of the leaky model in `baseline_benchmarks`.

## Think for yourself: ship a slightly better but harder-to-explain model?
Only because it is explainable: SHAP (Task 14) gives each prediction reasons, its probabilities are calibrated, and the gain is small. I would ship v2 **as a support signal with the baseline kept as fallback**, and say plainly that the ranking gain is within noise.

## Interview questions
1. *How do you prove a model improvement is real?* Same folds, paired per-fold differences, compare the mean gain with the spread, then a locked test set once.
2. *Why calibrate?* So "0.20" means about 20 in 100, which HR can act on.
3. *How do you check for leakage?* Drop outcome columns in the pipeline, scan single-feature AUC, be suspicious of any jump.
