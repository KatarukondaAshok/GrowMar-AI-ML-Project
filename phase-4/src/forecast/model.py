"""Task 16: forecast headcount and monthly attrition for 6 months, with an 80% range, tested on the past.

Three candidates are compared by backtesting (hide the last months, forecast them, measure the error):
  baseline  : 'same as last year plus trend' (seasonal naive with drift); attrition rate: same-month average of 3 years
  ets       : damped-trend exponential smoothing (statsmodels), simulated 80% interval
  sarimax   : ARIMA with the workforce events as external inputs (hiring freeze, hiring drive, appraisal month)
Events are handled, not ignored: a freeze or a hiring drive moves headcount, so the model is told about it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
Z80 = 1.2816
HORIZON = 6
VERSION = "headcount-forecast-1.0.0"


def event_flags(months: pd.DatetimeIndex, dept: str, events: pd.DataFrame) -> pd.DataFrame:
    """freeze (+1 while hiring is frozen) and drive (+1 while a hiring drive runs). For the company total 'ALL',
    a drive in any department counts. Other events (policy, appraisal) are not hiring shocks and are ignored here."""
    f = pd.DataFrame(0.0, index=months, columns=["freeze", "drive"])
    for e in events.itertuples():
        applies = e.department_id == "ALL" or dept == "ALL" or e.department_id == dept
        if not applies:
            continue
        name = e.event_name.lower()
        kind = "freeze" if "freeze" in name else "drive" if any(w in name for w in ("hiring", "scale-up", "expansion", "build-out")) else None
        if kind:
            f.loc[(months >= pd.Timestamp(e.start_date)) & (months <= pd.Timestamp(e.end_date)), kind] = 1.0
    return f


def _future_months(last: pd.Timestamp, n: int = HORIZON) -> pd.DatetimeIndex:
    return pd.date_range(last + pd.offsets.MonthBegin(1), periods=n, freq="MS")


def baseline_forecast(y: pd.Series, kind: str = "level", n: int = HORIZON) -> pd.DataFrame:
    """level (headcount): value(t+k) = value(t+k-12) + (latest - value a year ago).  rate: mean of the same month in the last 3 years.
    The range comes from how wrong the same rule was at each horizon over the past 36 origins."""
    fut = _future_months(y.index[-1], n)

    def point(s: pd.Series, k: int) -> float:
        if kind == "level":
            return float(s.iloc[-12 + k - 1] + (s.iloc[-1] - s.iloc[-12])) if k <= 12 else float(s.iloc[-1])
        return float(np.mean([s.iloc[-12 * j + k - 1] for j in (1, 2, 3) if len(s) >= 12 * j]))
    pts = [point(y, k) for k in range(1, n + 1)]
    sig = []
    for k in range(1, n + 1):
        errs = [y.iloc[o + k] - point(y.iloc[: o + 1], k) for o in range(len(y) - 36 - n, len(y) - n) if o >= 36]
        sig.append(float(np.std(errs)) if errs else 1.0)
    pts, sig = np.array(pts), np.array(sig)
    return pd.DataFrame({"forecast": pts, "lower_80": pts - Z80 * sig, "upper_80": pts + Z80 * sig}, index=fut)


def ets_forecast(y: pd.Series, n: int = HORIZON, seasonal: bool = False) -> pd.DataFrame:
    fit = ExponentialSmoothing(y.astype(float), trend="add", damped_trend=True, seasonal="add" if seasonal else None,
                               seasonal_periods=12 if seasonal else None, initialization_method="estimated").fit()
    sims = fit.simulate(n, repetitions=500, error="add", random_state=42)
    return pd.DataFrame({"forecast": fit.forecast(n).values, "lower_80": np.quantile(sims, 0.10, axis=1),
                         "upper_80": np.quantile(sims, 0.90, axis=1)}, index=_future_months(y.index[-1], n))


def sarimax_forecast(y: pd.Series, exog: pd.DataFrame, exog_future: pd.DataFrame, n: int = HORIZON, order=(1, 1, 1)) -> pd.DataFrame:
    fit = SARIMAX(y.astype(float), exog=exog, order=order, trend="n").fit(disp=False)
    pred = fit.get_forecast(n, exog=exog_future).summary_frame(alpha=0.20)
    return pd.DataFrame({"forecast": pred["mean"].values, "lower_80": pred["mean_ci_lower"].values,
                         "upper_80": pred["mean_ci_upper"].values}, index=_future_months(y.index[-1], n))


def forecast(series: pd.Series, dept: str, events: pd.DataFrame, method: str, kind: str = "level",
             planned: pd.DataFrame | None = None, n: int = HORIZON) -> pd.DataFrame:
    """One forecast. `planned` can switch events on for the future months, e.g. a hiring drive that HR already announced."""
    if method == "baseline":
        return baseline_forecast(series, kind, n)
    if method == "ets":
        return ets_forecast(series, n, seasonal=(kind == "rate"))
    hist = event_flags(series.index, dept, events)
    fut_idx = _future_months(series.index[-1], n)
    fut = event_flags(fut_idx, dept, events)
    if planned is not None:
        fut = (fut + planned.reindex(fut_idx).fillna(0)).clip(upper=1)
    return sarimax_forecast(series, hist, fut, n)


def backtest(series: pd.Series, dept: str, events: pd.DataFrame, method: str, kind: str, origins: int = 3, n: int = HORIZON) -> pd.DataFrame:
    """Rolling-origin backtest: hide the last n months (and the n months before, etc.), forecast, compare with what happened."""
    rows = []
    for o in range(origins):
        cut = len(series) - n * (o + 1)
        train, test = series.iloc[:cut], series.iloc[cut: cut + n]
        f = forecast(train, dept, events, method, kind, n=n)
        err = test.values - f["forecast"].values
        rows.append({"origin": train.index[-1].strftime("%Y-%m"), "mae": float(np.mean(np.abs(err))),
                     "mape_pct": float(np.mean(np.abs(err) / np.maximum(np.abs(test.values), 1e-9)) * 100),
                     "coverage_80": float(np.mean((test.values >= f["lower_80"].values) & (test.values <= f["upper_80"].values))),
                     "bias": float(np.mean(err))})
    return pd.DataFrame(rows)
