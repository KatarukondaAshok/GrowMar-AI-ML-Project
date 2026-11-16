import numpy as np
import pandas as pd
import pytest

from src.forecast.model import HORIZON, backtest, baseline_forecast, event_flags, forecast


def series(n=60, slope=1.0, seed=0):
    idx = pd.date_range("2021-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    return pd.Series(100 + slope * np.arange(n) + rng.normal(0, 1, n), index=idx)


EVENTS = pd.DataFrame([{"start_date": "2022-04-01", "end_date": "2022-06-30", "department_id": "ALL", "event_name": "Hiring freeze"},
                       {"start_date": "2023-01-01", "end_date": "2023-02-28", "department_id": "D01", "event_name": "Engineering hiring drive"},
                       {"start_date": "2023-07-01", "end_date": "2023-07-31", "department_id": "ALL", "event_name": "Hybrid work policy introduced"}])


def test_event_flags_freeze_drive_and_ignored_policy():
    m = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
    f = event_flags(m, "D01", EVENTS)
    assert f.loc["2022-05-01", "freeze"] == 1 and f.loc["2023-01-01", "drive"] == 1
    assert f.loc["2023-07-01"].sum() == 0                       # a policy change is not a hiring shock
    assert event_flags(m, "D02", EVENTS).loc["2023-01-01", "drive"] == 0     # drive belongs to D01 only
    assert event_flags(m, "ALL", EVENTS).loc["2023-01-01", "drive"] == 1     # company total sees every drive


@pytest.mark.parametrize("method", ["baseline", "ets", "sarimax"])
def test_forecast_shape_order_and_range(method):
    f = forecast(series(), "ALL", EVENTS, method, "level")
    assert len(f) == HORIZON and (f.lower_80 <= f.forecast).all() and (f.forecast <= f.upper_80).all()
    assert f.index[0] == pd.Timestamp("2026-01-01")             # starts the month after the last known month
    assert f.forecast.iloc[-1] > series().iloc[-1] - 5          # a rising series is not forecast to collapse


def test_range_widens_with_horizon_for_baseline():
    f = baseline_forecast(series(seed=3))
    assert (f.upper_80 - f.lower_80).iloc[-1] >= (f.upper_80 - f.lower_80).iloc[0] - 1e-9


def test_backtest_reports_error_and_coverage():
    bt = backtest(series(), "ALL", EVENTS, "ets", "level", origins=2)
    assert len(bt) == 2 and (bt.mae > 0).all() and bt.coverage_80.between(0, 1).all()


def test_saved_forecast_matches_sample_format():
    d = pd.read_csv("data/forecast_6_months.csv")
    assert list(d.columns) == ["forecast_id", "metric", "department_id", "forecast_month", "forecast_value", "lower_80", "upper_80", "method", "trained_through", "model_version"]
    assert len(d) == 72 and (d.lower_80 <= d.forecast_value).all() and (d.forecast_value <= d.upper_80).all()
    assert d.groupby(["metric", "department_id"]).size().eq(6).all()
