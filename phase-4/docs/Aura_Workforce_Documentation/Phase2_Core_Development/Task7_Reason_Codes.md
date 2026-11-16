# Task 7: Reason Codes and Money at Risk

**Phase 2 (Core Development) | Status: Done (friend test pending) | Code:** `src/payroll/reasons.py` | **Folder:** `phase-2/Task7_Reason_Codes`

## 1. Objective
Give every flag a reason code, a one-sentence reason a clerk understands, and the rupees at risk, in the official output format, ranked by money.

## 2. Why it matters
Unexplained alerts get ignored. A clear reason and a rupee amount makes people act. A reason that says "the model said so" is not accepted in an audit.

## 3. Input data
The flags from Task 6 and the sheet `sample_anomaly_results`, which defines the official columns.

## 4. What was done, step by step
1. Studied the official sample's columns.
2. Defined a reason code and sentence for every kind of flag (table below).
3. Computed `money_at_risk` = |net pay paid minus net pay expected|, where expected net = expected gross times the employee's usual net/gross ratio. Special cases: duplicate = the whole second net pay; missing PF = the usual PF amount; arithmetic mismatch = the unexplained gap.
4. Returned results in exactly the sample's 12 columns: `result_id, source_dataset, record_id, employee_id, period, anomaly_score, flagged, reason_code, reason_text, money_at_risk, model_version, scored_at`.
5. Ranked flags by money at risk.

## 5. Reason codes
| Code | Sentence a clerk reads |
|---|---|
| DUPLICATE_PAYMENT | Second payslip paid to the same employee for the same month |
| ZERO_PAY | Active employee shows zero pay |
| NEGATIVE_NET_PAY | Deductions exceed gross pay |
| GROSS_ABOVE_NORM | Gross is 10.0x what the fixed pay and days worked allow |
| GROSS_BELOW_EXPECTED | Gross is 12% below what fixed pay and days worked allow |
| OVERTIME_ABOVE_CAP | 96 paid overtime hours exceeds the 50-hour cap |
| PF_NOT_DEDUCTED | Provident fund missing although deducted in other months |
| ROLE_PAY_ABOVE_PEERS | Fixed pay is 2.9x the usual pay for this role |
| NET_PAY_MISMATCH | Net pay does not equal gross minus deductions |
| STATISTICAL_OUTLIER | Unusual combination of pay ratios versus all other payslips |

A payslip gets the highest-priority rule that fired. Only when no rule fired does it get `STATISTICAL_OUTLIER`. Reasons are written first, models second.

## 6. Deliverables
`flags_20_rows.csv` (top 20), `payroll_flags_official_format.csv` (all 223), `reason_codes.csv`, `FRIEND_FEEDBACK.md`.

## 7. Results
223 flags in the official 12-column format; the top flag is about Rs 21 lakh.

## 8. Think for yourself: two flags with the same reason
The bigger rupee amount goes first. If amounts are equal: higher anomaly score, then employee id, then record id, so the output is identical on every run. A clerk has limited time, so the list should start with the mistakes that cost the most.

## 9. Mini challenge: how a flagged payslip travels
1. Raw row `PI-003304` arrives: gross 28,23,640 against fixed pay 1,97,044.
2. `gross_vs_expected = 10.0`; the rule `GROSS_ABOVE_NORM` fires, score 1.0.
3. The reason code and sentence are written, and money at risk is Rs 21,02,520.
4. It ranks first of 223 flags.
5. `POST /v1/anomaly/payroll` returns it in the official format.

## 10. Still to do (by you)
**Friend test:** show a friend the 20 rows in `flags_20_rows.csv`, hide the code column, ask what each means, and record their answers in `FRIEND_FEEDBACK.md`.

## 11. Interview questions
1. **Why is "the model said so" not accepted in an audit?** An auditor needs a checkable reason and amount.
2. **How do you estimate money at risk?** The gap between paid and expected take-home pay, with documented special cases.
3. **How do you prioritise alerts?** By rupee impact, with a deterministic tie-break.
