import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Task 1 - Data Audit & Exploration
**Project:** Aura Workforce / CortexHR attrition  |  **Dataset:** `attrition_dataset` sheet  |  **Author:** Katarukonda Ashok

This notebook profiles the attrition table before any modelling: structure, duplicates, missing values, outliers (two methods), class imbalance and data-leakage risks. Every issue ends with a written decision, which becomes a rule in the Task 2 pipeline.

Run top to bottom (`Kernel > Restart & Run All`). The reusable functions live in `src/audit/checks.py`.""")

code("""import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
# Find the project root (the folder that contains `src/`) no matter where the notebook is opened from
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists())
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display
from src.audit import checks as ck

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
OUT = ROOT / "Task1_Data_Audit" / "outputs"; FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
raw = ck.load_attrition(DATA)
print("Loaded:", raw.shape)""")

md("## 1. Structure: rows, columns, types")
code("""print(f"Rows: {len(raw):,}   Columns: {raw.shape[1]}")
print("Sources:", raw["source"].value_counts().to_dict())
print("Observation dates:", raw.observation_date.min().date(), "to", raw.observation_date.max().date())
print("Unique employee_id:", raw.employee_id.nunique(), "(rows:", len(raw), ")")
types = raw.dtypes.astype(str).value_counts()
display(types.rename("columns").to_frame())""")

md("""### Data dictionary (my own words)
| Group | Columns | Meaning |
|---|---|---|
| Identity / bookkeeping | `row_id`, `employee_id`, `observation_date`, `source` | Row counter, employee code, the date the features were measured, and whether the row is from the real 30-Sep-2025 snapshot or the historical extension |
| Job & location | `department_id`, `role_id`, `job_level` (1-7), `work_location`, `work_mode`, `employment_type`, `highest_education` | Where and how the person works |
| Experience | `tenure_months`, `total_experience_years`, `previous_employers_count`, `commute_distance_km` | Time at this company, total career length, job hopping, daily travel |
| Pay | `annual_fixed_pay` (INR), `comp_ratio`, `raises_received`, `last_raise_pct`, `months_since_last_raise` | Salary and how it compares to the role midpoint (1.0 = at midpoint) |
| Career | `promotions_received`, `months_since_last_promotion`, `internal_transfers`, `last_performance_rating` (1-5), `last_engagement_score` (1-5), `manager_changes_12m`, `training_hours_12m` | Growth, review and satisfaction signals |
| Attendance (last 3-6 months) | `attendance_rate_3m`, `late_arrivals_3m`, `avg_overtime_hours_per_month_3m`, `avg_hours_worked_3m`, `wfh_share_3m`, `unpaid_leave_days_6m`, `paid_leave_days_6m` | How regularly the person shows up and how much they work |
| Context | `dept_attrition_rate_prior_12m` | Share of the department that left in the 12 months **before** the snapshot |
| **Target** | `left_within_12m` | 1 = left within 12 months after `observation_date`, 0 = stayed |
| Label-derived | `voluntary_left_within_12m` | 1 = left voluntarily (a subset of the target) |
| Post-exit (known only after leaving) | `exit_date`, `resignation_date`, `exit_reason`, `exit_interview_done`, `fnf_settlement_amount` | Exit paperwork. These are **leakage** |""")

md("## 2. Duplicates")
code("""dups = ck.find_exact_duplicates(raw)
display(dups[["row_id", "employee_id", "source", "tenure_months", "annual_fixed_pay"]])
print(f"{len(dups)//2} employees appear twice (identical in every column except row_id).")
df = ck.drop_exact_duplicates(raw)
print("After removing duplicates:", df.shape, "| employee_id unique:", df.employee_id.is_unique)""")
md("""**Decision D1 - drop the 6 exact duplicate rows (keep the first).** If one copy lands in train and the other in test, the model sees the same person twice and scores look better than they really are.""")

md("## 3. Missing values")
code("""mt = ck.missing_table(df)
mt.to_csv(TAB / "missing_values.csv")
display(mt)
fig, ax = plt.subplots(figsize=(7, 4))
mt["percent"].sort_values().plot.barh(ax=ax, color="#3b6ea8")
ax.set_xlabel("% of rows missing"); ax.set_title("Missing values per column")
plt.tight_layout(); plt.savefig(FIG / "missing_values.png"); plt.close()""")

md("### Are the gaps real data loss or 'not applicable'?")
code("""sm = ck.structural_missing_checks(df)
display(sm)
display(ck.missing_vs_target(df, ["last_engagement_score", "last_performance_rating", "last_raise_pct", "commute_distance_km"]))""")
md("""**What this shows**
- `months_since_last_promotion` (61.0% missing) and `last_raise_pct` (8.4%) are blank **exactly** when the person has never been promoted / never got a raise. The gap is meaningful, not lost data.
- `last_performance_rating` is blank for people who joined only a few months ago (average tenure of the blank group is about 5 months), so they have simply not been reviewed yet.
- `last_engagement_score` (22.2%) is blank when the person did not answer the survey, and those people leave slightly more often, so "missing" itself may carry signal.
- `commute_distance_km` (2.3%) and `highest_education` (0.6%) look like random small gaps.

**Decisions**
| Column | Missing | Decision |
|---|---|---|
| `months_since_last_promotion` | 61.0% | Fill with a sentinel (e.g. tenure in months) **and** keep `promotions_received`, or add flag `never_promoted`. Do **not** drop |
| `last_raise_pct` | 8.4% | Fill 0 and flag `no_raise_yet` |
| `last_performance_rating` | 7.6% | Flag `not_reviewed_yet`, then fill with median (fitted on train only) |
| `last_engagement_score` | 22.2% | Add flag `engagement_missing`, then median fill (train only) |
| `commute_distance_km` | 2.3% | Median fill by work_location (train only) |
| `highest_education` | 0.6% | Fill with `Unknown` category |
| 5 exit columns | 85-87% | **Drop** (leakage, see section 7) |

**Think for yourself: a column has 40% missing values. Drop, fill or flag?**
I do not drop first. I ask *why* it is missing. If the gap means something (like "never promoted") I fill it sensibly and add a "was missing" flag, because the missingness is itself information. I only drop a column when it is mostly empty **and** unrelated to the target, or when it leaks. Dropping the 61%-missing promotion column would throw away a real attrition signal.""")

md("## 4. Outliers: two methods compared")
code("""cols = ["annual_fixed_pay", "avg_hours_worked_3m", "tenure_months",
        "avg_overtime_hours_per_month_3m", "attendance_rate_3m", "commute_distance_km"]
oc = ck.outlier_compare(df, cols)
oc.to_csv(TAB / "outlier_comparison.csv", index=False)
display(oc)""")
code("""# Salary: raw vs log vs within job level
df["log_pay"] = np.log10(df.annual_fixed_pay)
print("IQR outliers  raw pay :", int(ck.iqr_outliers(df.annual_fixed_pay).sum()))
print("Z-score       raw pay :", int(ck.z_outliers(df.annual_fixed_pay).sum()))
print("IQR outliers  log pay :", int(ck.iqr_outliers(df.log_pay).sum()))
print("IQR outliers  within each job level :",
      sum(int(ck.iqr_outliers(g).sum()) for _, g in df.groupby("job_level").annual_fixed_pay))

fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
ax[0].boxplot(df.annual_fixed_pay / 1e5, vert=False); ax[0].set_title("Pay (lakh INR), raw"); ax[0].set_xlabel("lakh INR")
ax[1].boxplot(df.log_pay, vert=False); ax[1].set_title("Pay, log10 scale")
df.boxplot(column="annual_fixed_pay", by="job_level", ax=ax[2]); ax[2].set_title("Pay by job level"); ax[2].set_yscale("log")
plt.suptitle(""); plt.tight_layout(); plt.savefig(FIG / "salary_outliers.png"); plt.close()""")

code("""# Overtime: why IQR over-flags
ot = df.avg_overtime_hours_per_month_3m
print(f"Share of employees with zero overtime: {(ot == 0).mean():.0%}")
print("IQR flags every non-zero value because the IQR is 0:", int(ck.iqr_outliers(ot).sum()), "rows")
print("Overtime-eligible values range:", ot[ot > 0].min(), "to", ot[ot > 0].max())
print()
print("Lowest hours worked / attendance:")
display(df.nsmallest(3, "avg_hours_worked_3m")[["employee_id", "avg_hours_worked_3m", "attendance_rate_3m", "unpaid_leave_days_6m", ck.TARGET]])""")

md("""**IQR vs z-score: what I learned**
| Column | Reading |
|---|---|
| `annual_fixed_pay` | Right-skewed (skew 4.8). IQR flags 93 and z-score 26, but most are just senior people. On a log scale or within each job level almost nothing is flagged (1-9). Only **3 rows are real errors**: pay is ~10x too high and no longer matches `comp_ratio` (section 5) |
| `avg_overtime_hours_per_month_3m` | 81% of people have 0 overtime, so IQR = 0 and it flags every overtime worker (445). That is a false alarm. Use eligibility-aware rules instead |
| `avg_hours_worked_3m` | One genuine extreme row (EMP-0313: 1.8 hours, 20% attendance). Most other flags are mild |
| `tenure_months` | Long tenures (up to 164 months) are real veterans; z-score flags 45 but they are valid. The **4 negative values** are errors |
| `attendance_rate_3m` | Skewed toward 1.0. 2 impossible values above 1 (section 5) |

**Decisions**
- D2: Do **not** remove high pay by IQR or z-score. Fix the 3 typos, then use `log(pay)` or `comp_ratio` for modelling.
- D3: Winsorise or cap skewed counts (late arrivals, overtime, leave) at the 99th percentile, fitted on train only, rather than deleting rows.
- D4: Keep EMP-0313 but flag it for HR to verify (long absence or bad record) and cap hours at a sensible range.
- D5: Leave tenure outliers (genuine veterans) alone.""")

md("## 5. Impossible values and dirty categories")
code("""iv = ck.impossible_values(df); display(iv); iv.to_csv(TAB / "impossible_values.csv", index=False)
print("Salary typos (pay vs role midpoint implied by comp_ratio):")
display(ck.salary_typos(df))
print("Negative tenure:"); display(df[df.tenure_months < 0][["employee_id", "tenure_months", "total_experience_years"]])
print("Attendance above 1:"); display(df[df.attendance_rate_3m > 1][["employee_id", "attendance_rate_3m"]])
print("Messy work_location spellings:"); display(ck.messy_categories(df))
print("wfh_share_3m by work_mode:"); display(df.groupby("work_mode").wfh_share_3m.describe().round(3))
print("Negative total experience by source:", df[df.total_experience_years < 0].source.value_counts().to_dict())""")
md("""**Findings and decisions**
- D6: **3 salary typos** (EMP-0034, EMP-0433, EMP-0601) are 10x too high. Correct to `annual_fixed_pay / 10` (consistent with the role midpoint) and confirm with the Python team.
- D7: **4 negative tenures** (EMP-0248, 0280, 0568, 0596): the observation date is before the join date. Set tenure to 0 and flag for source-system check.
- D8: **2 rows with attendance above 1** (1.12): clip to 1.0.
- D9: **work_location spellings** (13 rows such as `BENGALURU`, `Bangalore`, `Madras`, `delhi ncr`): standardise with `ck.clean_location`.
- D10: **`wfh_share_3m` outside 0-1** (190 slightly negative on Onsite, 23 slightly above 1 on Remote): looks like small random noise added to the share. Clip to [0, 1].
- D11: **10 historical rows with negative `total_experience_years`** (-0.1 to -1.1): clip at 0.
- D20: **90 historical rows where tenure is more than 6 months longer than total experience** (up to 2.7 years; all from the historical extension). Experience cannot be shorter than time at this company, so set `total_experience_years = max(total_experience_years, tenure_months / 12)`.""")

md("## 6. Class imbalance (the attrition label)")
code("""cb = ck.class_balance(df); print(cb)
pd.Series(cb).to_csv(TAB / "class_balance.csv", header=["value"])
by_dept = df.groupby("department_id")[ck.TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
display(by_dept.round(3))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].bar(["Stayed", "Left"], [cb["stayed"], cb["left"]], color=["#7aa6c2", "#d9694a"])
for i, v in enumerate([cb["stayed"], cb["left"]]): ax[0].text(i, v + 20, f"{v:,}", ha="center")
ax[0].set_title(f"Class balance: {cb['left_pct']}% left")
by_dept["mean"].sort_values().plot.barh(ax=ax[1], color="#3b6ea8"); ax[1].axvline(cb["left_pct"]/100, color="k", ls="--", lw=1)
ax[1].set_title("Attrition rate by department"); ax[1].set_xlabel("share who left")
plt.tight_layout(); plt.savefig(FIG / "class_balance.png"); plt.close()""")
md("""**Finding:** only **14.6%** left (349 of 2,383), about **5.8 stayers per leaver**. A model that always says "stays" is **85.4% accurate** and catches nobody, so accuracy is a misleading metric here.

**Decisions**
- D12: Report **precision, recall, F1 and ROC-AUC** (and PR-AUC), not accuracy.
- D13: Use **stratified** splits and 5-fold stratified CV, and try `class_weight="balanced"` first. Use SMOTE only inside CV folds if needed.
- D14: Department D06 is far above average (21%) and D10 far below (6%), so keep `department_id` as a feature and check per-department performance.""")

md("## 7. Data leakage risks")
code("""lk = ck.leakage_table(df); display(lk); lk.to_csv(TAB / "leakage_columns.csv", index=False)""")
code("""from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

num_clean = ["tenure_months","comp_ratio","months_since_last_raise","manager_changes_12m","commute_distance_km",
             "last_engagement_score","last_performance_rating","attendance_rate_3m","late_arrivals_3m",
             "avg_overtime_hours_per_month_3m","unpaid_leave_days_6m","dept_attrition_rate_prior_12m"]
X_clean = df[num_clean].fillna(df[num_clean].median())
X_leak = X_clean.copy()
X_leak["has_exit_date"] = df.exit_date.notna().astype(int)
X_leak["settlement"] = df.fnf_settlement_amount.fillna(0)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
auc_clean = cross_val_score(mk(), X_clean, df[ck.TARGET], cv=cv, scoring="roc_auc")
auc_leak  = cross_val_score(mk(), X_leak,  df[ck.TARGET], cv=cv, scoring="roc_auc")
print(f"ROC-AUC with clean features : {auc_clean.mean():.3f} +/- {auc_clean.std():.3f}")
print(f"ROC-AUC with exit columns   : {auc_leak.mean():.3f} +/- {auc_leak.std():.3f}   <- too good to be true")

auc1 = ck.single_feature_auc(df, [c for c in df.select_dtypes("number").columns
                                  if c not in ["row_id", ck.TARGET, "voluntary_left_within_12m", "fnf_settlement_amount"]])
display(auc1.head(8).to_frame("single-feature AUC"))
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(["Clean features", "With exit columns"], [auc_clean.mean(), auc_leak.mean()], color=["#3b6ea8", "#d9694a"])
for i, v in enumerate([auc_clean.mean(), auc_leak.mean()]): ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
ax.set_ylim(0, 1.1); ax.set_ylabel("ROC-AUC (5-fold CV)"); ax.set_title("Leakage makes a model look perfect")
plt.tight_layout(); plt.savefig(FIG / "leakage_demo.png"); plt.close()""")
md("""**Leakage list (drop before any modelling)**
| Column | Why it leaks |
|---|---|
| `exit_date`, `resignation_date` | Only exist for people who left |
| `exit_reason`, `exit_interview_done` | Collected during exit |
| `fnf_settlement_amount` | Full-and-final payout, calculated at exit |
| `voluntary_left_within_12m` | A subset of the target itself |
| `row_id`, `employee_id` | Identifiers, no real signal and risk of memorising people |
| `source` | Records how the row was created (real snapshot vs extension) |

**Safe but check:** `dept_attrition_rate_prior_12m` is built from the 12 months *before* the snapshot (legitimate). `comp_ratio`, `months_since_last_raise` and the attendance windows are measured on or before the observation date (legitimate).

**Honest signal check:** the strongest single clean feature (`last_engagement_score`) reaches only AUC ~0.61, and the whole clean model about 0.70. That is realistic. If a later model suddenly scores 0.95+, suspect leakage first.

**Decisions**
- D15: Drop the 5 exit columns + `voluntary_left_within_12m` in the pipeline (Task 2).
- D16: Drop `row_id`, `employee_id`, `source` from model inputs. Keep `observation_date` only to build time-based splits.
- D17: Fit imputers, scalers and encoders on training data only.
- D18: Rows span 2021-2025, so check a time-based split in addition to random stratified CV.
- D19: `protected_attributes` (gender, age) is a separate sheet. Do not use it as a feature without a fairness review.""")

md("## 8. Decision log (for the report and Task 2)")
code("""decisions = pd.DataFrame([
 ("D1","Duplicates","6 exact duplicate rows","Drop (keep first)"),
 ("D2","Outliers","Pay is right-skewed; IQR/z flag senior staff","Keep rows; model log(pay) / comp_ratio"),
 ("D3","Outliers","Skewed counts (late, overtime, leave)","Cap at 99th percentile (train only)"),
 ("D4","Outliers","EMP-0313: 1.8 hrs, 20% attendance","Keep, flag for HR check"),
 ("D5","Outliers","Long tenures up to 164 months","Genuine; keep"),
 ("D6","Errors","3 salaries 10x too high","Divide by 10, confirm with Python team"),
 ("D7","Errors","4 negative tenures","Set to 0, flag source check"),
 ("D8","Errors","2 attendance rates = 1.12","Clip to 1.0"),
 ("D9","Errors","13 inconsistent work_location spellings","Standardise"),
 ("D10","Errors","wfh_share outside 0-1","Clip to [0,1]"),
 ("D11","Errors","10 negative total experience","Clip at 0"),
 ("D20","Errors","90 rows: tenure > total experience by 6+ months","Raise experience to tenure"),
 ("D12","Imbalance","14.6% left","Use precision/recall/F1/ROC-AUC, not accuracy"),
 ("D13","Imbalance","5.8:1 ratio","Stratified CV + class weights"),
 ("D14","Imbalance","Dept D06 21% vs D10 6%","Keep department; check per-dept results"),
 ("D15","Leakage","5 exit columns + voluntary label","Drop"),
 ("D16","Leakage","IDs and source","Exclude from model inputs"),
 ("D17","Leakage","Preprocessing fitted on all data","Fit on train only"),
 ("D18","Leakage","Dates span 2021-2025","Also test a time-based split"),
 ("D19","Fairness","Gender and age in separate sheet","Do not use without review"),
 ("M1","Missing","months_since_last_promotion 61% (never promoted)","Sentinel + never_promoted flag"),
 ("M2","Missing","last_raise_pct 8.5% (no raise)","Fill 0 + flag"),
 ("M3","Missing","last_performance_rating 7.6% (new joiners)","Flag + median"),
 ("M4","Missing","last_engagement_score 22%","Flag + median"),
 ("M5","Missing","commute_distance_km 2.3%","Median by location"),
 ("M6","Missing","highest_education 0.6%","'Unknown' category"),
], columns=["id","area","issue","decision"])
decisions.to_csv(TAB / "decision_log.csv", index=False)
display(decisions)""")
md("""## 9. Summary
The table is mostly usable but needs cleaning before modelling. Found independently: 6 exact duplicates, 4 negative tenures, 3 salary typos (10x), 2 attendance rates above 1, 13 inconsistent location spellings, about 70 random gaps (commute, education), plus extras such as noisy `wfh_share_3m`, 10 negative experience values, 90 rows with tenure longer than experience and 22% missing engagement. The label is imbalanced (14.6% left), and five exit columns plus the voluntary label leak the answer. Every item has a written decision above, and these flow into the Task 2 pipeline.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, "Task1_Data_Audit/task1_data_audit.ipynb")
print("written")
