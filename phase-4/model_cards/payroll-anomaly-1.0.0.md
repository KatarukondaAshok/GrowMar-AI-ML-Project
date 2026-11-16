# Model card: Payroll anomaly detector (`payroll-anomaly-1.0.0`)

| Field | Value |
|---|---|
| Version | payroll-anomaly-1.0.0 |
| Owner | Katarukonda Ashok (GM00154), AI/ML Engineer, Aura Workforce (CortexHR) |
| Status | Released as v1.0.0 for review; a support signal for people, never a decision |

## Purpose
Flags unusual payslips before payment, with a reason code and the rupees at risk.

## Intended use and out-of-scope use
- **Use:** A payroll clerk or finance reviewer checks flagged payslips before a run is paid.
- **Do not use for:** Never to pay, withhold or recover money automatically, and never to judge a person.

## Data
`payroll_monthly_features`: 7,355 payslips, 528 employees. 100 injected errors (8 types) used ONLY to measure, never to train. All data is fictional (Zenith Nova Technologies).

## Method
Readable rules (duplicate, zero pay, negative net, gross far from the pay formula, overtime above 50 hours, PF missing, pay far above role peers, arithmetic mismatch) plus an Isolation Forest on 12 ratio features (top 2.5%). A flag is any rule or a model flag; the highest-priority rule gives the reason.

## Metrics
Rules + model: recall 0.98 (98 of 100), precision 0.44. Rules alone: precision 1.00, recall 0.93. Reference baseline: precision 0.22, recall 0.40.

## Fairness
No personal attribute is a feature. Flags depend on pay arithmetic and history only.

## Limits (honest)
Errors are synthetic, so real recall will be lower. The model adds little over the rules (5 of 130 model-only flags were real). Combined precision 0.44 means more than half of flags are false alarms.

## Monitoring
Re-check flag rate per run; a jump means data or pay-structure change.
