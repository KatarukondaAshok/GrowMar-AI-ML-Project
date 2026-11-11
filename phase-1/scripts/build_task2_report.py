import json
from reportkit import *

T = "Task2_Feature_Pipeline/"
R = json.load(open(T + "outputs/results.json"))
FOOT = "Task 2 - Reusable Feature Pipeline | Aura Workforce attrition | Katarukonda Ashok (GM00154)"
s = [P("Task 2: Reusable Feature Pipeline", h1),
     P("Documentation | Pipeline version <b>features-1.0.0</b> | Prepared by Katarukonda Ashok (GM00154) | October 2026", sub)]

s += [P("1. What this task is, in plain words", h2),
      P("Raw HR data is messy, and every model needs it cleaned the same way. Think of the pipeline as a <b>recipe card</b> written once and reused by every model: "
        "it repairs bad values, builds new features, fills gaps, scales numbers and encodes categories. Because the recipe is saved and versioned, "
        "the live service will prepare data exactly as training did, and model scores can be compared fairly."),
      P("<b>One rule governs the whole task:</b> the pipeline may only <i>learn</i> (medians, caps, category lists, department pay levels) from the "
        "<b>training rows</b>. Learning from all rows before splitting lets test information leak into training.")]

s += [P("2. What the pipeline does, step by step", h2)]
s += [tbl([
    ["Step", "Class (file src/features/)", "What it does", "Learns from train?"],
    ["1 clean", "RowCleaner (transformers.py)", "Fixes city spellings, negative tenure, attendance above 1, wfh_share outside 0-1, negative experience, experience shorter than tenure", "No (fixed rules)"],
    ["2 fix_salary", "SalaryTypoFixer", "Finds pay about 10x too high by comparing it with the role's usual pay band, then divides by 10", "Yes: pay band of each role"],
    ["3 engineer", "FeatureEngineer", "Adds flags (never_promoted, no_raise_yet, not_reviewed_yet, engagement_missing) and new features; fills not-applicable gaps", "Yes: department median pay"],
    ["4 encode", "ColumnTransformer (pipeline.py)", "Numbers: cap at 1st/99th percentile, median fill, scale. Flags: pass through. Categories: fill Unknown, one-hot. Everything else is dropped", "Yes: caps, medians, scales, categories"],
], [20 * mm, 38 * mm, W - 98 * mm, 40 * mm])]

s += [P("Features requested by the task", h2),
      tbl([["Requested", "Column(s) in the output"],
           ["Attendance rate", "attendance_rate_3m, attendance_gap (1 minus rate)"],
           ["Late count", "late_arrivals_3m"],
           ["Average overtime", "avg_overtime_hours_per_month_3m, overtime_per_year_of_tenure"],
           ["Tenure", "tenure_months, tenure_years"],
           ["Salary versus department median", "pay_vs_dept_median (pay divided by the department median learned on train), log_pay"],
           ["Months since last revision", "months_since_last_raise (never-raised people are flagged by no_raise_yet)"]], [52 * mm, W - 52 * mm])]

s += [keep(P("3. Audit decisions turned into code", h2),
           tbl([["Task 1 decision", "Where it lives"],
                ["D1 duplicates", "data.drop_duplicates, before the split"],
                ["D3 cap skewed values", "Winsorizer"],
                ["D6 salary typos", "SalaryTypoFixer"],
                ["D7-D11, D20 impossible values", "RowCleaner"],
                ["M1-M4 'not applicable' gaps", "FeatureEngineer flags plus sensible fills"],
                ["M5-M6 small random gaps", "median fill / category Unknown"],
                ["D15-D16 leakage and identifiers", "never selected by the ColumnTransformer (remainder='drop')"],
                ["D17 fit on train only", "fit_pipeline(train) and cloning inside cross-validation"]], [52 * mm, W - 52 * mm]))]

s += [P("4. Results", h2)]
g = R["column_groups"]
s += [tbl([
    ["Check", "Result"],
    ["Rows", f"{R['raw_rows']:,} raw, {R['rows_after_dedup']:,} after removing duplicates; split into {R['train_rows']:,} train and {R['test_rows']:,} test (leaver rate {R['train_leaver_rate']:.1%} and {R['test_leaver_rate']:.1%}: stratified)"],
    ["Output columns", f"{R['output_columns']} = {g['numeric (scaled)']} numeric + {g['flags (0/1)']} flags + {g['categories (one-hot)']} one-hot categories"],
    ["Missing values after transform", f"{R['missing_after_transform']}"],
    ["Leakage / identifier columns in output", f"{R['leakage_columns_in_output']}"],
    ["Salary typos fixed in training rows", f"{R['salary_typos_fixed_in_training']} (the third typo row fell in the test part and is fixed there using train-learned pay bands)"],
    ["Save, load, transform", f"Loaded pipeline gives identical output: {R['loaded_matches_original']}"],
    ["Version in metadata", R["version"]],
    ["Automated tests", R["tests_summary"].replace("\x1b", "")],
    ["One-command run (Docker/CLI)", "Works" if R["cli_ok"] else "Failed"],
], [52 * mm, W - 52 * mm])]
s += [Spacer(1, 3), P("<b>Why train-only fitting matters (example):</b> department median pay computed from train and from all rows differ slightly. "
        "If the all-rows version were used, the test rows would influence the features the model trains on. The notebook (Step 7) prints both side by side.")]

s += [P("5. How to run it", h2),
      P("The files live in <font name='Courier'>src/features/</font> (config, data, transformers, pipeline, cli). From the project root:"),
      C("python -m src.features.cli run --input data/raw/02_CortexHR_AI_ML_Dataset.xlsx --output-dir artifacts\n"
        "python -m src.features.cli transform --model artifacts/features-1.0.0.joblib --input new_rows.csv --output features.csv"),
      P("The first command removes duplicates, splits, fits on train, saves <font name='Courier'>features-1.0.0.joblib</font> plus a metadata file, and writes "
        "<font name='Courier'>X_train.csv</font> and the locked test files. Docker version:"),
      C("docker build -t aura-attrition .\ndocker run --rm -v \"$PWD:/app\" aura-attrition python -m src.features.cli run --output-dir artifacts"),
      P("Use it in Python:"),
      C("from src.features.pipeline import fit_pipeline, save_pipeline, load_pipeline, transform_dataframe\n"
        "pipe = fit_pipeline(train_df)                       # train rows only\n"
        "save_pipeline(pipe, train_df, 'artifacts')\n"
        "pipe, meta = load_pipeline('artifacts/features-1.0.0.joblib')\n"
        "X = transform_dataframe(pipe, new_rows_df)")]

s += [keep(P("6. Completion criteria", h2),
           checklist([("Pipeline fits, saves, loads and transforms", "Done", "Step 8 of the notebook; identical output after reload"),
                      ("Version number recorded", "Done", f"{R['version']} inside the saved file and the .meta.json"),
                      ("Tests pass", "Done", "tests/test_features.py: output shape, no missing values, same input same output, save/load, no leakage, train-only fitting"),
                      ("Runs inside Docker / from the command line", "Done*", "CLI verified; *Dockerfile provided (see README to build)")]))]
s += [P("<b>Next:</b> Task 3 plugs this pipeline in front of each baseline model and refits it inside every cross-validation fold, so even the medians and caps are learned from training folds only.")]
build(T + "Task2_Documentation.pdf", s, "Task 2 - Reusable Feature Pipeline", FOOT)
