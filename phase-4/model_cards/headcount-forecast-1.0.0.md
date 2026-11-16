# Model card: Headcount and attrition forecast (`headcount-forecast-1.0.0`)

| Field | Value |
|---|---|
| Version | headcount-forecast-1.0.0 |
| Owner | Katarukonda Ashok (GM00154), AI/ML Engineer, Aura Workforce (CortexHR) |
| Status | Released as v1.0.0 for review; a support signal for people, never a decision |

## Purpose
Forecasts headcount (company and 10 departments) and the monthly attrition rate for 6 months, with an 80% range.

## Intended use and out-of-scope use
- **Use:** HR and finance planning of hiring and budgets.
- **Do not use for:** Not a promise: always show the range. Not for individual decisions.

## Data
`monthly_headcount_history`: 81 months, 12 series; `workforce_events`: 9 events. All data is fictional (Zenith Nova Technologies).

## Method
ARIMA(1,1,1) with hiring freeze and hiring-drive inputs for headcount; damped-trend exponential smoothing for the attrition rate; baseline 'same as last year plus trend'.

## Metrics
3 origins x 6 months: headcount MAE 2.33 heads (baseline 2.78); attrition rate MAE 0.51 points (baseline 0.53). Strict last-6-months test on the company total: baseline 5.3, ARIMA 7.3, so the better model is not clearly better there. The 80% range held 69% to 73% of the time.

## Fairness
Works on counts, not people, so no personal attributes are used.

## Limits (honest)
Ranges are slightly too narrow. The monthly attrition rate is noisy (error about 68%). Unknown future events are assumed to be absent.

## Monitoring
Compare each month's actual headcount with the forecast and its range.
