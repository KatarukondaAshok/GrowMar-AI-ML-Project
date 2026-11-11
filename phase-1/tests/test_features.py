"""Task 2 tests: shape, no missing values, same input -> same output, save/load, no leakage, train-only fitting."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_VERSION
from src.features.config import CATEGORICAL, LEAKAGE_COLUMNS, NUMERIC_BASE, TARGET
from src.features.data import drop_duplicates, split_train_test
from src.features.pipeline import (build_feature_pipeline, fit_pipeline, load_pipeline, output_columns,
                                   save_pipeline, transform_dataframe)
from src.features.transformers import FeatureEngineer, RowCleaner, SalaryTypoFixer, Winsorizer

DATA = Path("data/raw/02_CortexHR_AI_ML_Dataset.xlsx")


def make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: rng.normal(10, 3, n) for c in NUMERIC_BASE})
    d["job_level"] = rng.integers(1, 6, n)
    d["tenure_months"] = rng.integers(1, 120, n).astype(float)
    d["total_experience_years"] = d["tenure_months"] / 12 + rng.uniform(0, 5, n)
    d["attendance_rate_3m"] = rng.uniform(0.7, 1.0, n)
    d["wfh_share_3m"] = rng.uniform(0, 1, n)
    d["comp_ratio"] = rng.uniform(0.8, 1.2, n)
    d["role_id"] = rng.choice(["R1", "R2", "R3"], n)
    d["annual_fixed_pay"] = d["comp_ratio"] * d["role_id"].map({"R1": 5e5, "R2": 8e5, "R3": 12e5})
    d["department_id"] = rng.choice(["D01", "D02", "D03"], n)
    d["work_location"] = rng.choice(["Bengaluru", "BENGALURU", "Madras", "Pune"], n)
    d["work_mode"] = rng.choice(["Onsite", "Hybrid", "Remote"], n)
    d["employment_type"] = rng.choice(["Permanent", "Fixed-Term Contract"], n)
    d["highest_education"] = rng.choice(["B.Tech", "MBA", None], n)
    for c in ["months_since_last_promotion", "last_raise_pct", "last_performance_rating", "last_engagement_score", "commute_distance_km"]:
        d.loc[rng.choice(n, 40, replace=False), c] = np.nan
    for c in LEAKAGE_COLUMNS:
        d[c] = rng.integers(0, 2, n)
    d["employee_id"] = [f"E{i}" for i in range(n)]
    d["row_id"] = range(n)
    d["source"] = "x"
    d[TARGET] = rng.integers(0, 2, n)
    return d


@pytest.fixture(scope="module")
def df():
    return make_df()


@pytest.fixture(scope="module")
def fitted(df):
    train = df.iloc[:200]
    return train, fit_pipeline(train)


def test_output_shape_matches_column_list(df, fitted):
    train, pipe = fitted
    out = transform_dataframe(pipe, df.iloc[200:])
    assert out.shape == (100, len(output_columns(pipe)))


def test_no_missing_values_after_transform(df, fitted):
    _, pipe = fitted
    assert transform_dataframe(pipe, df).isna().sum().sum() == 0


def test_same_input_same_output(df, fitted):
    _, pipe = fitted
    a, b = transform_dataframe(pipe, df), transform_dataframe(pipe, df)
    pd.testing.assert_frame_equal(a, b)


def test_two_fits_on_same_data_agree(df):
    train = df.iloc[:200]
    a = transform_dataframe(fit_pipeline(train), df.iloc[200:])
    b = transform_dataframe(fit_pipeline(train), df.iloc[200:])
    pd.testing.assert_frame_equal(a, b)


def test_save_load_gives_identical_output_and_version(df, fitted, tmp_path):
    train, pipe = fitted
    path = save_pipeline(pipe, train, tmp_path)
    loaded, meta = load_pipeline(path)
    assert meta["version"] == FEATURE_VERSION == "features-1.0.0"
    assert meta["n_training_rows"] == 200
    pd.testing.assert_frame_equal(transform_dataframe(pipe, df), transform_dataframe(loaded, df))
    assert (tmp_path / f"{FEATURE_VERSION}.meta.json").exists()


def test_leakage_and_id_columns_never_reach_the_model(fitted):
    _, pipe = fitted
    cols = output_columns(pipe)
    for bad in LEAKAGE_COLUMNS + ["employee_id", "row_id", "source", "observation_date", "role_id"]:
        assert not any(c == bad or c.startswith(bad + "_") for c in cols)


def test_fitting_uses_training_rows_only(df):
    train = df.iloc[:200]
    fe = FeatureEngineer().fit(train)
    assert fe.dept_median_ == train.groupby("department_id")["annual_fixed_pay"].median().to_dict()
    assert fe.dept_median_ != df.groupby("department_id")["annual_fixed_pay"].median().to_dict()


def test_unseen_category_does_not_break(df, fitted):
    _, pipe = fitted
    new = df.iloc[:5].copy()
    new["work_location"] = "Atlantis"
    new["department_id"] = "D99"
    assert transform_dataframe(pipe, new).isna().sum().sum() == 0


def test_row_cleaner_fixes_impossible_values():
    d = pd.DataFrame({"work_location": ["BENGALURU", "Madras"], "tenure_months": [-3.0, 24.0],
                      "attendance_rate_3m": [1.12, 0.9], "wfh_share_3m": [-0.05, 1.04],
                      "total_experience_years": [-0.3, 0.5], "highest_education": [None, "MBA"]})
    out = RowCleaner().fit_transform(d)
    assert list(out.work_location) == ["Bengaluru", "Chennai"]
    assert out.tenure_months.min() >= 0 and out.attendance_rate_3m.max() <= 1
    assert out.wfh_share_3m.between(0, 1).all()
    assert out.total_experience_years.iloc[1] >= 2.0  # raised to tenure (24 months)
    assert out.highest_education.iloc[0] == "Unknown"


def test_salary_typo_fixer_divides_tenfold_error():
    d = pd.DataFrame({"role_id": ["R1"] * 6, "comp_ratio": [1.0] * 6,
                      "annual_fixed_pay": [500000, 510000, 490000, 505000, 495000, 5000000]})
    fixer = SalaryTypoFixer().fit(d)
    out = fixer.transform(d)
    assert out.annual_fixed_pay.iloc[-1] == 500000 and fixer.n_fixed_last_ == 1


def test_salary_fixer_handles_integer_pay_column():
    d = pd.DataFrame({"role_id": ["R1"] * 6, "comp_ratio": [1.0] * 6,
                      "annual_fixed_pay": np.array([500000, 510000, 490000, 505000, 495000, 3999999], dtype="int64")})
    out = SalaryTypoFixer().fit(d).transform(d)  # 399999.9 is not a whole number: must not crash
    assert abs(out.annual_fixed_pay.iloc[-1] - 399999.9) < 1e-6


def test_winsorizer_caps_extremes_and_keeps_nan():
    x = np.append(np.arange(100.0), [10_000.0]).reshape(-1, 1)
    w = Winsorizer(0.01, 0.99).fit(x)
    out = w.transform(np.array([[10_000.0], [np.nan], [50.0]]))
    assert out[0, 0] < 200 and np.isnan(out[1, 0]) and out[2, 0] == 50.0


def test_pipeline_is_cloneable_for_cross_validation():
    from sklearn.base import clone
    assert clone(build_feature_pipeline()) is not None


@pytest.mark.skipif(not DATA.exists(), reason="dataset not in data/raw")
def test_real_dataset_end_to_end(tmp_path):
    from src.features.data import load_clean_base
    full = load_clean_base(str(DATA))
    assert full.employee_id.is_unique
    train, test = split_train_test(full)
    pipe = fit_pipeline(train)
    X = transform_dataframe(pipe, test)
    assert X.isna().sum().sum() == 0 and len(X) == len(test)
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.01  # stratified


def test_dedup_removes_exact_copies():
    d = pd.DataFrame({"row_id": [1, 2, 3], "employee_id": ["A", "A", "B"], "x": [1, 1, 2]})
    assert len(drop_duplicates(d)) == 2
