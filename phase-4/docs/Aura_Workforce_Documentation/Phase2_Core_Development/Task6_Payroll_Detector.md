# Task 6: Payroll Outlier Detector

**Phase 2 (Core Development) | Status: Done | Code:** `src/payroll/features.py`, `rules.py`, `evaluate.py` | **Tests:** `tests/test_payroll_rules.py` | **Folder:** `phase-2/Task6_Payroll_Detector`

## 1. Objective
Catch unusual payslips before payment, and measure the detector separately for each of the 8 error types, not only overall.

## 2. Why it matters
A payroll error hits a real person or costs the company money. Catching it before payment is worth far more than finding it later.

## 3. Input data
Sheet `payroll_monthly_features`: 7,355 payslips, 23 columns. The sheet `payroll_anomaly_labels` holds 100 injected errors and is used only to measure, never to train or tune.

## 4. What was done, step by step
1. **Read the columns** and wrote what each one means.
2. **Learned "normal" without labels:** median fixed pay per role, each person's usual PF and net/gross ratio, and which employee-months were already paid.
3. **Built ratio features.** A ratio is "this payslip divided by what it should be", so 1.0 means normal. Main ratios: `gross_vs_expected` (10 means a slipped decimal), `ded_ratio` (deductions / gross, normally about 5%), `net_ratio` (negative means the person owes money), `ot_hours` (overtime hours, cap 50), `fixed_vs_role` (a clerk paid like a manager is 3), `pf_vs_own` (PF stopped being deducted).
4. **Wrote rules,** each one readable line: duplicate payment, zero pay, negative net, gross far above or below expected, overtime above 50 hours, PF missing though paid before, fixed pay over 2 times role peers, arithmetic mismatch.
5. **Isolation Forest** on 12 ratios, flagging the top 2.5%.
6. **Combined:** flag = any rule OR model flag. Rules give the reason; the model catches what no rule describes.
7. **Measured with labels afterwards,** per error type.

## 5. Deliverables
`src/payroll/` code, per-error-type table (`per_error_type_results.csv`), recall chart, `rule_examples.csv`.

## 6. Results (100 injected errors in 7,355 payslips)
| Detector | Flagged | Caught | Precision | Recall |
|---|---|---|---|---|
| Reference baseline | 184 | 40 | 0.217 | 0.40 |
| Model alone | 184 | 59 | 0.321 | 0.59 |
| Rules alone | 93 | 93 | 1.00 | 0.93 |
| **Rules + model** | 223 | 98 | 0.439 | **0.98** |

Per type, rules alone catch 100% of double payments, wrong decimals, missed deductions, role mismatches, stale salaries, negative net and zero pay. Overtime spikes reach 87% combined.

## 7. What the rules caught that the model missed
All 15 double payments and all 12 stale salaries. Each row looks normal on its own; only context (the same employee paid twice, or pay below the formula) reveals them.

## 8. Think for yourself: which type is hardest?
For the model alone: double payment and stale salary (0%). For the whole system: overtime spikes (87%), because 7 of 15 are about 8 times normal but still under the 50-hour cap. A person finds duplicates easy because they see the same name twice in a list; a model scores one row at a time.

## 9. Interview questions
1. **Why combine rules and a model?** Rules are exact and explainable for known mistakes; the model finds unknown patterns. Finance teams trust a rule they can read.
2. **How do you evaluate an unsupervised detector?** With hidden injected labels, measured per error type, never used to fit.
3. **Why ratios instead of raw amounts?** A 5,00,000 payslip is normal for a CEO and an error for a clerk.
4. **Precision or recall here?** Missing an error costs money, but too many false alarms make people ignore the list, so rules (precision 1.0) come first.

## 10. Honest notes
Rules do the work: of 130 model-only flags, only 5 are injected errors. The combined precision is 0.44. Injected errors are synthetic, so real recall will be lower.
