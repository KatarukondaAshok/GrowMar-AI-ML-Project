# Model card: Pay band recommender (`paybands-v1`)

| Field | Value |
|---|---|
| Version | paybands-v1 |
| Owner | Katarukonda Ashok (GM00154), AI/ML Engineer, Aura Workforce (CortexHR) |
| Status | Released as v1.0.0 for review; a support signal for people, never a decision |

## Purpose
Recommends a low, mid and high yearly pay for a new offer.

## Intended use and out-of-scope use
- **Use:** A manager and HR use the range as advice when preparing an offer.
- **Do not use for:** Never to set an individual's pay automatically.

## Data
`salary_training`: 2,934 hires (job-related fields only). All data is fictional (Zenith Nova Technologies).

## Method
Ridge regression on log(pay / band mid-point); range from the 10th and 90th percentile of out-of-fold errors; clipped to the role's band.

## Metrics
Test MAE Rs 54,195 vs Rs 115,094 for the role average; R2 0.983; 80.7% of true offers fall inside the range.

## Fairness
Gender and career gap are never inputs. History shows women were paid 1.1 points of band mid-point less, and people with 7 to 12 months of career gap 2.1 points less; the final model does not copy this. The unexplained shortfall is reported to HR.

## Limits (honest)
Other features can still correlate with gender or career gaps. Only 60 people in 'Other'. Not causal proof.

## Monitoring
Pay-gap check every week (Task 18): latest WATCH at 1.68 points for career gap.
