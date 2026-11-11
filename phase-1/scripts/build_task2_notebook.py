from nbkit import NB, setup

nb = NB()
nb.md("""# Task 2 - Reusable Feature Pipeline
**Project:** Aura Workforce attrition | **Author:** Katarukonda Ashok | **Pipeline version:** `features-1.0.0`

**Goal:** turn the decisions from the Task 1 audit into ONE packaged pipeline that every later model reuses. It must fit once, save, load later and transform new rows in exactly the same way.

**Why:** if every model prepares data its own way, scores cannot be compared and the live service will not match training.

**Rule of the whole task:** the pipeline learns (medians, caps, categories, department pay) from the **training rows only**. Learning from the whole dataset before splitting is leakage.

Run top to bottom (`Run All`). The code lives in `src/features/`; this notebook explains and proves it.""")

nb.code(setup("Task2_Feature_Pipeline"))
nb.code("""from src.features import FEATURE_VERSION, __version__
from src.features import config as cfg
from src.features.data import load_raw, load_clean_base, split_train_test
from src.features.pipeline import (build_feature_pipeline, fit_pipeline, transform_dataframe, output_columns,
                                   save_pipeline, load_pipeline)
from src.features.transformers import RowCleaner, SalaryTypoFixer, FeatureEngineer, Winsorizer
RESULTS = {}
print("Package version:", __version__, "| feature version:", FEATURE_VERSION)""")

nb.md("""## Step 1. Load, de-duplicate, split BEFORE fitting
Duplicates (decision D1) are removed first so one person cannot sit in both train and test. Then a **stratified** 80/20 split keeps the 14.6% leaver rate in both parts. The test part stays locked for Task 3.""")
nb.code("""raw = load_raw(DATA)
df = load_clean_base(DATA)
train, test = split_train_test(df)
print(f"raw rows {len(raw):,} -> after removing duplicates {len(df):,}")
print(f"train {len(train):,} rows (leavers {train[cfg.TARGET].mean():.2%})   |   test {len(test):,} rows (leavers {test[cfg.TARGET].mean():.2%})")
RESULTS.update(raw_rows=len(raw), rows_after_dedup=len(df), train_rows=len(train), test_rows=len(test),
               train_leaver_rate=float(train[cfg.TARGET].mean()), test_leaver_rate=float(test[cfg.TARGET].mean()))""")

nb.md("""## Step 2. From audit decisions to code
Every decision in the Task 1 log has a home in the pipeline.

| Decision | Problem (Task 1) | Where it is handled |
|---|---|---|
| D1 | 6 duplicate rows | `data.drop_duplicates` (before the split) |
| D6 | 3 salaries 10x too high | `SalaryTypoFixer` (learns each role's usual pay band) |
| D7, D8, D9, D10, D11, D20 | negative tenure, attendance above 1, city spellings, wfh outside 0-1, negative experience, tenure above experience | `RowCleaner` (fixed rules, learns nothing) |
| D3 | skewed counts and pay | `Winsorizer` (caps at 1st/99th percentile learned on train) |
| M1-M4 | "not applicable" gaps | `FeatureEngineer` creates `never_promoted`, `no_raise_yet`, `not_reviewed_yet`, `engagement_missing` flags and fills the gaps sensibly |
| M5-M6 | small random gaps | median fill (`SimpleImputer`) and the category `Unknown` |
| D15, D16 | leakage columns, IDs | dropped by the final `ColumnTransformer` (`remainder="drop"`) |
| D17 | fit on train only | every step learns in `fit()`; the pipeline is fitted with `fit_pipeline(train)` |

**Engineered features asked for in the task:** attendance rate (`attendance_rate_3m`, plus `attendance_gap`), late count (`late_arrivals_3m`), average overtime (`avg_overtime_hours_per_month_3m`, `overtime_per_year_of_tenure`), tenure (`tenure_months`, `tenure_years`), salary versus department median (`pay_vs_dept_median`), months since last revision (`months_since_last_raise`).""")

nb.md("## Step 3. The pipeline object")
nb.code("""pipe = build_feature_pipeline()
for i, (name, step) in enumerate(pipe.steps, 1):
    print(f"{i}. {name:11s} -> {type(step).__name__}")
print()
enc = pipe.named_steps["encode"]
for name, trans, cols in enc.transformers:
    t = trans if isinstance(trans, str) else " -> ".join(type(s).__name__ for _, s in trans.steps)
    print(f"   encode/{name:5s}: {t:55s} {len(cols)} columns")""")

nb.md("## Step 4. Fit on the TRAINING rows only and look at what it learned")
nb.code("""pipe = fit_pipeline(train)
fe, fixer = pipe.named_steps["engineer"], pipe.named_steps["fix_salary"]
print("Median pay per department (learned from train):")
display(pd.Series(fe.dept_median_).round(0).astype(int).rename("median pay (INR)").to_frame().T)
print("Role pay bands learned:", len(fixer.role_mid_), "roles | salary typos fixed in train:",
      (lambda c: (fixer.transform(c), fixer.n_fixed_last_)[1])(pipe.named_steps["clean"].transform(train)))
num = pipe.named_steps["encode"].named_transformers_["num"]
caps = pd.DataFrame({"low_cap": num.named_steps["cap"].low_, "high_cap": num.named_steps["cap"].high_}, index=cfg.NUMERIC_COLUMNS)
display(caps.loc[["tenure_months", "late_arrivals_3m", "avg_overtime_hours_per_month_3m", "pay_vs_dept_median"]].round(2))""")

nb.md("## Step 5. Transform: shape, missing values, column list")
nb.code("""X_train = transform_dataframe(pipe, train)
X_test = transform_dataframe(pipe, test)
cols = output_columns(pipe)
groups = {"numeric (scaled)": [c for c in cols if c in cfg.NUMERIC_COLUMNS],
          "flags (0/1)": [c for c in cols if c in cfg.FLAG_COLUMNS],
          "categories (one-hot)": [c for c in cols if c not in cfg.NUMERIC_COLUMNS + cfg.FLAG_COLUMNS]}
print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Missing values after transform: train", int(X_train.isna().sum().sum()), "| test", int(X_test.isna().sum().sum()))
display(pd.Series({k: len(v) for k, v in groups.items()}, name="columns").to_frame())
display(X_train.iloc[:3, :8].round(3))
leaks = [c for c in cols if any(c == b or c.startswith(b + "_") for b in cfg.LEAKAGE_COLUMNS + cfg.ID_COLUMNS + ["role_id", "observation_date"])]
print("Leakage / identifier columns in the output:", leaks or "none")
RESULTS.update(output_columns=len(cols), missing_after_transform=int(X_train.isna().sum().sum() + X_test.isna().sum().sum()),
               column_groups={k: len(v) for k, v in groups.items()}, leakage_columns_in_output=len(leaks))""")

nb.md("""## Step 6. See the clean-up working (before and after)
These are the exact problem rows found in Task 1.""")
nb.code("""cleaner = RowCleaner().fit(df)
before = df[df.employee_id.isin(["EMP-0248", "EMP-0280"])][["employee_id", "tenure_months", "total_experience_years", "work_location", "attendance_rate_3m"]]
after = cleaner.transform(df[df.employee_id.isin(["EMP-0248", "EMP-0280"])])[before.columns]
print("Negative tenure: BEFORE"); display(before)
print("AFTER RowCleaner"); display(after)

fx = SalaryTypoFixer().fit(df)
typos = df[df.employee_id.isin(["EMP-0034", "EMP-0433", "EMP-0601"])][["employee_id", "role_id", "annual_fixed_pay", "comp_ratio"]]
fixed = fx.transform(typos)
out = typos.assign(pay_after_fix=fixed.annual_fixed_pay)
print("Salary typos (fitted on the full table here only to demo; real use fits on train):"); display(out)

wf = df.copy(); wf["wfh_share_3m"].describe()
print("wfh_share_3m range  before:", round(df.wfh_share_3m.min(), 3), "to", round(df.wfh_share_3m.max(), 3),
      "| after:", cleaner.transform(df).wfh_share_3m.min(), "to", cleaner.transform(df).wfh_share_3m.max())
print("work_location values  before:", df.work_location.nunique(), "| after:", cleaner.transform(df).work_location.nunique())""")

nb.md("""## Step 7. Why "fit on train only" matters (a concrete example)
`pay_vs_dept_median` divides pay by the department median. If the median were computed on ALL rows, the test rows would influence the numbers the model trains on. The two sets of medians are close but not equal, and that gap is exactly the leakage we avoid.""")
nb.code("""m_train = train.groupby("department_id").annual_fixed_pay.median()
m_all = df.groupby("department_id").annual_fixed_pay.median()
cmp = pd.DataFrame({"median_from_train": m_train, "median_from_all_rows": m_all})
cmp["difference"] = cmp.median_from_train - cmp.median_from_all_rows
display(cmp.round(0).astype(int).T)
print("Learned by the saved pipeline == train medians:", fe.dept_median_ == m_train.to_dict())""")

nb.md("## Step 8. Save with a version, load later, get the same answer")
nb.code("""art = OUT / "artifacts"
path = save_pipeline(pipe, train, art, extra={"n_test_rows_locked": len(test)})
print("Saved:", path.relative_to(ROOT), f"({path.stat().st_size/1024:.0f} KB)")
loaded, meta = load_pipeline(path)
same = transform_dataframe(pipe, test).equals(transform_dataframe(loaded, test))
print("Version in metadata:", meta["version"], "| rows it was trained on:", meta["n_training_rows"], "| output columns:", meta["n_output_columns"])
print("Loaded pipeline gives identical output to the original:", same)
print("Same input twice -> same output:", transform_dataframe(loaded, test).equals(transform_dataframe(loaded, test)))
display(pd.Series({k: meta[k] for k in ["version", "created_utc", "random_seed", "sklearn_version", "n_training_rows", "n_output_columns", "salary_typos_fixed_in_training"]}, name="metadata").to_frame())
RESULTS.update(version=meta["version"], loaded_matches_original=bool(same), salary_typos_fixed_in_training=meta["salary_typos_fixed_in_training"])""")

nb.md("## Step 9. Run the tests")
nb.code("""import subprocess
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_features.py", "-q", "--no-header", "--color=no"], cwd=ROOT, capture_output=True, text=True)
print(r.stdout[-1500:])
RESULTS["tests_summary"] = r.stdout.strip().splitlines()[-1]""")

nb.md("## Step 10. One command from raw data to model-ready features")
nb.code("""import subprocess
cmd = [sys.executable, "-m", "src.features.cli", "run", "--input", str(DATA), "--output-dir", str(OUT / "cli_demo")]
print("$ python -m src.features.cli run --input data/raw/02_CortexHR_AI_ML_Dataset.xlsx --output-dir artifacts")
r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
print(r.stdout)
print(sorted(p.name for p in (OUT / "cli_demo").iterdir()))
RESULTS["cli_ok"] = r.returncode == 0""")

nb.md("""## Summary
- The pipeline cleans, engineers, caps, fills, scales and encodes in one object (`features-1.0.0`).
- It is fitted on the training rows only, saved with `joblib`, and gives identical output after loading.
- Leakage columns and identifiers never reach the model; no missing values remain after the transform.
- Task 3 plugs this exact pipeline in front of each model and refits it inside every cross-validation fold.""")
nb.code("""(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
import shutil; shutil.rmtree(OUT / "cli_demo", ignore_errors=True)   # demo files only; keep the folder tidy
print(json.dumps(RESULTS, indent=2, default=str)[:900])""")

nb.save("Task2_Feature_Pipeline/task2_feature_pipeline.ipynb")
