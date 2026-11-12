# Task 6: Payroll outlier detector

**Goal:** flag unusual payslips and measure the detector for each of the 8 error types. **Result:** rules + Isolation Forest catch 98 of 100 injected errors.

## How it works (step by step)
1. **Read the columns** of `payroll_monthly_features` (23 columns, 7,355 payslips; see the data dictionary sheet).
2. **Learn "normal" without labels** (`src/payroll/features.py::fit_reference`): median fixed pay per role, each person's usual PF and net/gross ratio, and which employee-months were already paid.
3. **Ratio features** (`build_features`). A ratio is "this payslip divided by what it should be", so 1.0 means normal:

| Ratio | Meaning | Real-world example |
|---|---|---|
| `gross_vs_expected` | gross pay / (fixed pay x days paid / days in month + overtime + variable) | Salary 50,000 for a full month but gross shows 5,00,000 -> ratio 10 (decimal slipped) |
| `gross_vs_fixed` | gross / fixed monthly gross | Same idea, ignoring part-month pay |
| `ded_ratio` | total deductions / gross | Normal ~5%. 130% means deductions exceed pay |
| `net_ratio` | net / gross | Normal ~0.93. Negative means the person owes money |
| `tds_ratio` | tax / gross | A typo in tax shows up here |
| `ded_sum_gap`, `net_gap` | do the parts add up? | Net must equal gross minus deductions |
| `ot_hours`, `ot_rate_ratio` | overtime hours; overtime pay vs 2x hourly rate | 96 hours when the cap is 50 |
| `fixed_vs_role` | fixed pay / median of the same role | A clerk paid like a manager (3x) |
| `fixed_vs_own` | fixed pay / this person's usual | Sudden jump or old salary after a raise |
| `pf_vs_own` | PF this month / this person's usual PF | PF stopped being deducted |

4. **Rules** (`src/payroll/rules.py`): duplicate payment, zero pay, negative net, gross far above or below expected, overtime above 50 hours, PF missing though paid before, fixed pay over 2x role peers, arithmetic mismatch. Each rule is one readable line. Thresholds come from business logic and the *unlabelled* data (normal payslips match the gross formula exactly, so a 5% tolerance is generous).
5. **Isolation Forest** on the 12 ratios, flagging the top 2.5% (same flag budget as the reference baseline, chosen without labels).
6. **Combine:** flag = any rule OR model flag. Rules give the reason; the model catches what no rule describes.
7. **Measure with labels afterwards** (`src/payroll/evaluate.py`). Labels are never used to train or tune.

## Results (100 injected errors, 7,355 payslips)
| Detector | Flagged | Caught | Precision | Recall |
|---|---|---|---|---|
| Reference baseline | 184 | 40 | 0.217 | 0.40 |
| Model alone (my ratios) | 184 | 59 | 0.321 | 0.59 |
| Rules alone | 93 | 93 | **1.00** | 0.93 |
| **Rules + model** | 223 | 98 | 0.439 | **0.98** |

Per error type (`outputs/tables/per_error_type_results.csv`, figure `outputs/figures/recall_by_error_type.png`):

| Error type | Injected | Model alone | Rules alone | Combined |
|---|---|---|---|---|
| DOUBLE_PAYMENT | 15 | 0% | 100% | 100% |
| WRONG_DECIMAL | 15 | 100% | 100% | 100% |
| MISSED_DEDUCTION | 15 | 40% | 100% | 100% |
| ROLE_MISMATCH | 15 | 100% | 100% | 100% |
| OVERTIME_SPIKE | 15 | 87% | 53% | 87% |
| STALE_SALARY | 12 | 0% | 100% | 100% |
| NEGATIVE_NET | 8 | 100% | 100% | 100% |
| ZERO_PAY | 5 | 40% | 100% | 100% |

Every rule has **precision 1.0** against its target type (no false alarms from rules).

## What the rules caught that the model missed (vs `anomaly_baselines`)
- **All 15 double payments and all 12 stale salaries:** each row looks perfectly normal on its own. Only context (the same employee paid twice, or pay below the formula) reveals them.
- 9 of 15 missed deductions and 3 of 5 zero-pay rows: easy for a rule, diluted for a forest.
Examples with item ids: `outputs/tables/rule_examples.csv`.

## Think for yourself
**Hardest type?** For the model alone, DOUBLE_PAYMENT and STALE_SALARY (0%). For the whole system, OVERTIME_SPIKE (87%): 7 of 15 spikes are about 8x normal but still under the 50-hour cap, so only a comparison with that person's own overtime history helps. A person finds duplicates easy because they look at a list and see the same name twice; a model scores one row at a time and has no "list" in its head.

## Honest notes
- **Rules do the work.** Of 130 model-only flags, only 5 are injected errors. The model's extra value is small here; it is a safety net, and those flags are low-priority leads.
- **Precision of the combined detector is 0.44** because the model's 2.5% budget adds 125 false alarms. A smaller model budget would raise precision but I did not tune it on labels.
- I read the label sheet's notes while planning, which hinted at the error kinds. The thresholds, however, come from payroll arithmetic and unlabelled distributions, and no label touches a model.
- Injected errors are synthetic, so real-world recall will be lower.

## Interview questions (practise out loud)
1. *Why combine rules and a model?* Rules are exact and explainable for known mistakes; the model finds unknown patterns. Finance teams trust a rule they can read.
2. *How do you evaluate an unsupervised detector?* With hidden injected labels, measured per error type, never used to fit.
3. *Why ratios instead of raw amounts?* A 5,00,000 payslip is normal for a CEO and an error for a clerk. A ratio compares each payslip with its own expected value.
4. *Precision vs recall here?* Missing an error costs money, but too many false alarms make people ignore the list, so rules (precision 1.0) come first.

Run: `python scripts/train_phase2_models.py` then `python scripts/run_phase2_evidence.py`. Tests: `tests/test_payroll_rules.py`.
