# Task 15: Recommend fair pay bands
Code: `src/pay/model.py`. Saved model: `models/paybands-v1.joblib`. Evidence: `outputs/tables/`, `outputs/figures/career_gap_gap.png`.

## Steps
1. Features (job-related only): job level, band mid-point, prior experience, skills count, market index, hire year, department, location, education. **Not used:** gender, age, career gap.
2. The model predicts pay as band mid-point x ratio, then three models are compared: role average (the brief's baseline), Ridge, LightGBM (5-fold CV, then a locked 20% test used once).
3. Fairness check with `group_gap()` by gender and career gap (unlock: `protected_attributes`).
4. Output a **range**: low and high come from the 10th and 90th percentile of out-of-fold errors, so about 80% of true offers fall inside (measured on the locked test: **80.7%**). For a new offer, today's market index applies and the range is kept inside the role's band.

## Accuracy
| Model | CV MAE (rupees) | CV R2 | Test MAE | Test R2 |
|---|---|---|---|---|
| Baseline: average pay of the role | 115,964 | 0.924 | 115,094 | 0.933 |
| **Linear (Ridge), job-related** | **54,293** | 0.982 | 54,195 | 0.983 |
| LightGBM, job-related | 54,412 | 0.981 | | |
Ridge is chosen: equally accurate and easier to explain.

## Fairness (gap vs the reference group, in points of the band mid-point)
| Group | Actual pay in the data | Model that sees gender and career gap | Final model, predicted | Actual minus predicted |
|---|---|---|---|---|
| Female vs male | -1.10 | -1.14 | +0.10 | -1.19 |
| Career gap 1-6 months | -2.45 | -2.47 | -1.31 | -1.14 |
| Career gap 7-12 months | -2.14 | -2.18 | -0.30 | -1.84 |
| Career gap over 12 months | -1.47 | -1.42 | +1.65 | -3.13 |
People with a career gap were paid less: ratio 0.796 vs 0.816 (p = 0.007; 385 vs 2,549 people). A model allowed to see gender and career gap **copies this history**. The final model does not predict the gap, and the last column shows the size of the historical shortfall that is *not* explained by the job (largest for gaps over 12 months).

## What the gap means and what I changed
Past pay contains a penalty for career breaks and a smaller one for women. I removed those features so the model does not copy it, then measured both predicted and actual-minus-predicted gaps. Remaining limits: other features correlate with gender/gap (so predicted gaps are not exactly zero), the data has only 60 people in "Other", and this is not causal proof.

## Think for yourself: should the model copy low pay for women?
No. It should predict pay for the **job**, not for the person's history. I would (1) exclude those features, (2) report the unexplained gap to HR so they can review existing pay, and (3) require a manager and HR to approve every offer; the model gives a range, not a decision.

## Interview questions
1. *Why a baseline?* To prove the model adds something: it halves the error (MAE 54k vs 116k).
2. *Why a range?* One number looks precise but is not; the range says how sure we are.
3. *How do you stop a model copying historical bias?* Remove the biased features, test predicted and residual gaps by group, review with HR.
