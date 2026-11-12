# Task 7: Reason codes and money at risk

**Goal:** every flag gets a reason code, a one-sentence reason a clerk understands and the rupees at risk, in the **official output format** (`sample_anomaly_results`), ranked by money. Code: `src/payroll/reasons.py`.

## Output columns (identical to the sample sheet)
`result_id, source_dataset, record_id, employee_id, period, anomaly_score, flagged, reason_code, reason_text, money_at_risk, model_version, scored_at`

Files: `outputs/tables/flags_20_rows.csv` (top 20), `payroll_flags_official_format.csv` (all 223), `reason_codes.csv`.

## Reason codes
| Code | Sentence a clerk reads |
|---|---|
| DUPLICATE_PAYMENT | Second payslip paid to the same employee for the same month |
| ZERO_PAY | Active employee shows zero pay |
| NEGATIVE_NET_PAY | Deductions exceed gross pay |
| GROSS_ABOVE_NORM | Gross is 10.0x what the fixed pay and days worked allow |
| GROSS_BELOW_EXPECTED | Gross is 12% below what the fixed pay and days worked allow |
| OVERTIME_ABOVE_CAP | 96 paid overtime hours exceeds the 50-hour cap |
| PF_NOT_DEDUCTED | Provident fund missing although it was deducted in other months |
| ROLE_PAY_ABOVE_PEERS | Fixed pay is 2.9x the usual pay for this role |
| NET_PAY_MISMATCH | Net pay does not equal gross pay minus deductions |
| STATISTICAL_OUTLIER | Unusual combination of pay ratios versus all other payslips |

Reasons are written first, models second: a payslip gets the **highest-priority rule** that fired; only when no rule fired does it get `STATISTICAL_OUTLIER`.

## Money at risk (rupees)
`money_at_risk = |net pay paid - net pay expected|`, where expected net = expected gross x this employee's usual net/gross ratio.
Special cases: duplicate = the whole second net pay; missing PF = the employee's usual PF amount; arithmetic mismatch = the unexplained gap. For `GROSS_ABOVE_NORM` this matches the sample sheet's figures within about 0.3%.

## Ranking and ties (think for yourself)
Sort by `money_at_risk` descending. **Two flags with the same reason:** the bigger rupee amount first. If amounts are equal, higher anomaly score, then employee id and record id, so output is identical on every run (tested). Reason: a clerk has limited time, so the list should start with the mistakes that cost the most.

## Mini challenge: how a flagged payslip travels (Phase 2, 5 lines)
1. Raw row `PI-003304` (EMP-0304, Dec 2025) arrives in `payroll_monthly_features`: gross 28,23,640 against fixed pay 1,97,044.
2. `build_features` computes `gross_vs_expected = 10.0`; the rule `GROSS_ABOVE_NORM` (limit 1.5) fires, so `anomaly_score = 1.0`.
3. `build_results` picks the reason code, writes "Gross is 10.0x what the fixed pay and days worked allow" and money at risk Rs 21,02,520.
4. The row is ranked first of 223 flags (largest money at risk).
5. `POST /v1/anomaly/payroll` returns it in the official format (see `Task8_Anomaly_API/outputs/curl_session.md`) and the HR chatbot can read it out.

## Friend test (TODO for you)
Show a friend the 20 rows in `flags_20_rows.csv`, hide the code column, and ask what each means. Record their answers in `FRIEND_FEEDBACK.md` (a template is provided). I cannot do this step for you.

## Interview questions
1. *Why is "the model said so" not accepted in an audit?* An auditor needs a checkable reason and amount; rules and ratios give both.
2. *How do you estimate money at risk?* Gap between paid and expected take-home pay, with documented special cases.
3. *How do you prioritise alerts?* By rupee impact, with a deterministic tie-break.
