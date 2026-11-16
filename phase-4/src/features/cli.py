"""Command line: one command turns raw exported data into model-ready features.

    python -m src.features.cli run --input data/raw/02_CortexHR_AI_ML_Dataset.xlsx --output-dir artifacts
    python -m src.features.cli transform --model artifacts/features-1.0.0.joblib --input new_rows.csv --output out.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import FEATURE_VERSION
from .config import DEFAULT_DATA, DEFAULT_SHEET, RANDOM_SEED, TARGET
from .data import load_clean_base, split_train_test
from .pipeline import fit_pipeline, load_pipeline, save_pipeline, transform_dataframe


def cmd_run(args) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_clean_base(args.input, args.sheet)
    train, test = split_train_test(df, seed=args.seed)
    pipe = fit_pipeline(train)                           # fitted on TRAIN only
    path = save_pipeline(pipe, train, out, extra={"n_test_rows_locked": int(len(test))})
    X_train, X_test = transform_dataframe(pipe, train), transform_dataframe(pipe, test)
    X_train.to_csv(out / "X_train.csv", index=False)
    X_test.to_csv(out / "X_test_LOCKED.csv", index=False)
    train[TARGET].to_csv(out / "y_train.csv", index=False)
    test[TARGET].to_csv(out / "y_test_LOCKED.csv", index=False)
    summary = {"version": FEATURE_VERSION, "rows_after_dedup": len(df), "train_rows": len(train),
               "test_rows": len(test), "feature_columns": X_train.shape[1],
               "missing_values_after_transform": int(X_train.isna().sum().sum() + X_test.isna().sum().sum()),
               "pipeline_file": str(path)}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_transform(args) -> None:
    pipe, meta = load_pipeline(args.model)
    src = Path(args.input)
    df = pd.read_csv(src) if src.suffix == ".csv" else pd.read_excel(src, sheet_name=args.sheet)
    feats = transform_dataframe(pipe, df)
    feats.to_csv(args.output, index=False)
    print(f"{meta['version']}: wrote {feats.shape[0]} rows x {feats.shape[1]} columns to {args.output}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="aura-features", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="clean, split, fit on train, save pipeline, write feature files")
    r.add_argument("--input", default=DEFAULT_DATA); r.add_argument("--sheet", default=DEFAULT_SHEET)
    r.add_argument("--output-dir", default="artifacts"); r.add_argument("--seed", type=int, default=RANDOM_SEED)
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("transform", help="apply a saved pipeline to new rows")
    t.add_argument("--model", required=True); t.add_argument("--input", required=True)
    t.add_argument("--output", required=True); t.add_argument("--sheet", default=DEFAULT_SHEET)
    t.set_defaults(func=cmd_transform)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
