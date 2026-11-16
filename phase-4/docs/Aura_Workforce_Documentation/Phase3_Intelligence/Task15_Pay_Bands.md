# Task 15: Recommend Fair Pay Bands

**Phase 3 (Intelligence) | Status: Done | Code:** `src/pay/model.py` | **Saved model:** `models/paybands-v1.joblib` | **Folder:** `phase-3/Task15_Pay_Bands`

## 1. Objective
Predict a sensible offered salary for a role, test the model for pay gaps between groups, and output a range instead of one number.

## 2. Why it matters
Pay is personal, and past pay can contain past bias. A model that copies history copies unfairness. A pay recommendation is advice; the decision stays with a manager and HR.

## 3. Input data
`salary_training` (2,934 hires) and `protected_attributes` (gender, used only to test fairness).

## 4. What was done, step by step
1. **Chose job-related features only:** job level, band mid-point, prior experience, skills count, market index, hire year, department, location, education. **Not used:** gender, age, career gap.
2. Compared three models: the role average (the baseline), Ridge, and LightGBM (5-fold CV, then a locked 20% test used once).
3. Ran the fairness check with `group_gap()` by gender and by career gap.
4. **Output a range:** low and high come from the 10th and 90th percentile of out-of-fold errors, so about 80% of true offers fall inside (measured on the locked test: **80.7%**). For a new offer, today's market index applies and the range stays inside the role's band.

## 5. Accuracy
| Model | CV MAE (rupees) | CV R2 | Test MAE | Test R2 |
|---|---|---|---|---|
| Baseline: average pay of the role | 115,964 | 0.924 | 115,094 | 0.933 |
| **Linear (Ridge), job-related** | **54,293** | 0.982 | 54,195 | 0.983 |
| LightGBM, job-related | 54,412 | 0.981 | | |

Ridge is chosen: equally accurate and easier to explain. It halves the baseline error.

## 6. Fairness (gap versus the reference group, in points of the band mid-point)
| Group | Actual pay in the data | Model that sees gender and career gap | Final model, predicted | Actual minus predicted |
|---|---|---|---|---|
| Female vs male | -1.10 | -1.14 | +0.10 | -1.19 |
| Career gap 1-6 months | -2.45 | -2.47 | -1.31 | -1.14 |
| Career gap 7-12 months | -2.14 | -2.18 | -0.30 | -1.84 |
| Career gap over 12 months | -1.47 | -1.42 | +1.65 | -3.13 |

People with a career gap were paid less (ratio 0.796 versus 0.816, p = 0.007). A model allowed to see gender and career gap copies this history. The final model does not predict the gap; the last column shows the historical shortfall that the job does not explain.

## 7. What the gap means and what I changed
Past pay contains a penalty for career breaks and a smaller one for women. I removed those features so the model does not copy it, then measured predicted and actual-minus-predicted gaps. Limits: other features correlate with gender and gap, there are only 60 people in "Other", and this is not causal proof.

## 8. Think for yourself: should the model copy low pay for women?
No. It should predict pay for the job, not for the person's history. Exclude those features, report the unexplained gap to HR so they can review existing pay, and require manager and HR approval for every offer.

## 9. Interview questions
1. **Why a baseline?** To prove the model adds something: it halves the error (MAE 54k versus 116k).
2. **Why a range?** One number looks precise but is not; the range says how sure we are.
3. **How do you stop a model copying historical bias?** Remove the biased features, test predicted and residual gaps by group, and review with HR.
