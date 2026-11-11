"""Task 4 tests: repeatable injection, valid injected rows, feature pipeline, threshold logic, save/load."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.anomaly.detect import (anomaly_scores, choose_threshold, load_detector, make_detector, save_detector,
                                sweep_thresholds)
from src.anomaly.inject import inject_anomalies
from src.features.attendance import FEATURE_COLUMNS, build_attendance_pipeline


def make_attendance(n_emp=60, months=12, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_emp):
        for m in range(months):
            sched = int(rng.integers(19, 23))
            pres = sched - int(rng.integers(0, 2))
            rows.append({"attendance_feature_id": f"AF-{e:03d}-{m:02d}", "employee_id": f"EMP-{e:04d}",
                         "month": pd.Timestamp(2025, 1, 1) + pd.DateOffset(months=m), "scheduled_days": sched,
                         "present_days": pres, "wfh_days": int(rng.integers(0, pres + 1)), "half_days_unpaid": 0,
                         "paid_leave_days": 0, "unpaid_absence_days": 0, "longest_unpaid_streak_days": 0,
                         "late_arrivals": int(rng.integers(0, 3)), "avg_hours_worked": round(float(rng.normal(8.5, 0.4)), 2),
                         "overtime_hours": 0.0, "attendance_rate": round(pres / sched, 4)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def att():
    return make_attendance()


@pytest.fixture(scope="module")
def injected(att):
    return inject_anomalies(att, {"LONG_ABSENCE": 10, "IMPOSSIBLE_HOURS": 10, "PATTERN_CHANGE": 10}, seed=42)


def test_injection_is_repeatable_with_fixed_seed(att):
    a = inject_anomalies(att, {"LONG_ABSENCE": 5, "IMPOSSIBLE_HOURS": 5, "PATTERN_CHANGE": 5}, seed=42)
    b = inject_anomalies(att, {"LONG_ABSENCE": 5, "IMPOSSIBLE_HOURS": 5, "PATTERN_CHANGE": 5}, seed=42)
    pd.testing.assert_frame_equal(a[0], b[0]); pd.testing.assert_frame_equal(a[1], b[1])
    c = inject_anomalies(att, {"LONG_ABSENCE": 5, "IMPOSSIBLE_HOURS": 5, "PATTERN_CHANGE": 5}, seed=7)
    assert not a[1].attendance_feature_id.equals(c[1].attendance_feature_id)


def test_injection_log_matches_flags_and_counts(injected):
    inj, log = injected
    assert inj.is_injected.sum() == len(log) == 30
    assert set(log.injection_type) == {"LONG_ABSENCE", "IMPOSSIBLE_HOURS", "PATTERN_CHANGE"}
    assert log.attendance_feature_id.is_unique


def test_injected_rows_stay_internally_valid(injected):
    inj, _ = injected
    assert (inj.present_days <= inj.scheduled_days).all() and (inj.present_days >= 0).all()
    assert inj.attendance_rate.between(0, 1).all()
    assert (inj.longest_unpaid_streak_days <= inj.unpaid_absence_days).all()


def test_injection_does_not_modify_the_input(att):
    before = att.copy()
    inject_anomalies(att, {"LONG_ABSENCE": 3, "IMPOSSIBLE_HOURS": 3, "PATTERN_CHANGE": 3}, seed=1)
    pd.testing.assert_frame_equal(before, att)


def test_feature_pipeline_shape_and_no_missing(att):
    X = build_attendance_pipeline().fit(att).transform(att)
    assert X.shape == (len(att), len(FEATURE_COLUMNS)) and not np.isnan(X).any()


def test_features_keep_row_order(att):
    shuffled = att.sample(frac=1, random_state=0)
    X = build_attendance_pipeline().fit(att).transform(shuffled)
    assert np.allclose(X[:, FEATURE_COLUMNS.index("attendance_rate")], shuffled.attendance_rate.values)


def test_detector_scores_injected_rows_higher(injected):
    inj, _ = injected
    pipe = make_detector(n_estimators=100).fit(inj)
    s = anomaly_scores(pipe, inj)
    assert s[inj.is_injected == 1].mean() > s[inj.is_injected == 0].mean()
    assert np.allclose(s, anomaly_scores(make_detector(n_estimators=100).fit(inj), inj))  # fixed seed -> same scores


def test_sweep_catch_rate_rises_with_flag_rate(injected):
    inj, _ = injected
    s = anomaly_scores(make_detector(n_estimators=100).fit(inj), inj)
    sw = sweep_thresholds(s, inj.is_injected.values)
    assert sw.catch_rate.is_monotonic_increasing and sw.false_alarm_rate.is_monotonic_increasing
    assert (sw.caught + sw.missed == inj.is_injected.sum()).all()


def test_choose_threshold_respects_flag_cap():
    sw = pd.DataFrame({"flag_rate": [0.01, 0.03, 0.05, 0.10], "f1": [0.2, 0.5, 0.45, 0.9]})
    assert choose_threshold(sw, max_flag_rate=0.05).flag_rate == 0.03


def test_save_and_load_detector(injected, tmp_path):
    inj, _ = injected
    pipe = make_detector(n_estimators=50).fit(inj)
    path = save_detector(pipe, 0.55, tmp_path, extra={"note": "test"})
    p2, thr, meta = load_detector(path)
    assert thr == 0.55 and meta["version"] == "attendance-anomaly-1.0.0" and meta["note"] == "test"
    assert np.allclose(anomaly_scores(pipe, inj), anomaly_scores(p2, inj))
