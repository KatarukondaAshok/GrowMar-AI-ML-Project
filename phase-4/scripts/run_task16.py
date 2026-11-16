"""Task 16: forecast headcount (company + 10 departments) and attrition rate for 6 months, backtest, chart, official format.
python scripts/run_task16.py"""
from __future__ import annotations

from _common import ROOT, out, save_json, workbook

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forecast.model import HORIZON, VERSION, backtest, forecast

OUT = out("Task16_Forecast")


def main() -> dict:
    x = pd.read_excel(workbook(), sheet_name=None)
    hist, events, sample = x["monthly_headcount_history"], x["workforce_events"], x["sample_forecast"]
    hist["month"] = pd.to_datetime(hist["month"])
    # (metric, dept, column, kind) series to forecast = exactly the 12 series in sample_forecast
    series = [("headcount_end", d, "headcount_end", "level") for d in ["ALL"] + [f"D{i:02d}" for i in range(1, 11)]]
    series.append(("attrition_rate_monthly_pct", "ALL", "attrition_rate_monthly_pct", "rate"))
    S = {(m, d): hist[hist.department_id == d].set_index("month")[c].astype(float) for m, d, c, k in series}
    trained_through = hist["month"].max().strftime("%Y-%m")

    # 1) Backtests: three origins x 6 months, all series, three methods
    rows = []
    for (m, d, c, k) in series:
        for method in ("baseline", "ets", "sarimax"):
            if k == "rate" and method == "sarimax":
                continue
            try:
                bt = backtest(S[(m, d)], d, events, method, k)
            except Exception:
                continue
            rows.append({"metric": m, "department_id": d, "method": method, **bt[["mae", "mape_pct", "coverage_80", "bias"]].mean().to_dict()})
    bt_all = pd.DataFrame(rows)
    bt_all.round(3).to_csv(OUT / "tables" / "backtest_all_series.csv", index=False)
    summary = bt_all.groupby(["metric", "method"]).agg(mae=("mae", "mean"), mape_pct=("mape_pct", "mean"), coverage_80=("coverage_80", "mean")).reset_index()
    summary.round(3).to_csv(OUT / "tables" / "backtest_summary.csv", index=False)
    best = {}
    for m in summary.metric.unique():
        t = summary[summary.metric == m].sort_values("mae").reset_index(drop=True)
        best[m] = t.iloc[0]["method"]
        ev = t[t.method == "sarimax"]
        # tie-break written in advance: if the event-aware model is within 5% of the best error, use it (it handles events and can take planned ones)
        if len(ev) and ev.iloc[0]["mae"] <= 1.05 * t.iloc[0]["mae"]:
            best[m] = "sarimax"

    # 2) The strict test the task asks for: hide the LAST 6 known months, forecast them, measure the error
    strict = []
    for (m, d, c, k) in series:
        s = S[(m, d)]
        for method in ("baseline", "ets", "sarimax"):
            if k == "rate" and method == "sarimax":
                continue
            try:
                f = forecast(s.iloc[:-HORIZON], d, events, method, k)
            except Exception:
                continue
            act = s.iloc[-HORIZON:].values
            strict.append({"metric": m, "department_id": d, "method": method, "mae": float(np.mean(np.abs(act - f.forecast.values))),
                           "mape_pct": float(np.mean(np.abs(act - f.forecast.values) / act) * 100),
                           "inside_80_range": int(((act >= f.lower_80.values) & (act <= f.upper_80.values)).sum())})
    st = pd.DataFrame(strict)
    st.round(3).to_csv(OUT / "tables" / "backtest_last_6_months.csv", index=False)

    # 3) Final forecast with the best method per metric; official sample columns
    fc_rows, i = [], 1
    labels = {"baseline": "seasonal naive with drift (same as last year plus trend)", "ets": "damped-trend exponential smoothing",
              "sarimax": "ARIMA(1,1,1) with workforce events as inputs"}
    for (m, d, c, k) in series:
        method = best[m]
        f = forecast(S[(m, d)], d, events, method, k)
        for mon, r in f.iterrows():
            fc_rows.append({"forecast_id": f"FC-{i:04d}", "metric": m, "department_id": d, "forecast_month": mon.strftime("%Y-%m-%d"),
                            "forecast_value": round(r.forecast, 1), "lower_80": round(min(r.lower_80, r.forecast), 1),
                            "upper_80": round(max(r.upper_80, r.forecast), 1), "method": labels[method], "trained_through": trained_through,
                            "model_version": f"{VERSION}"})
            i += 1
    fc = pd.DataFrame(fc_rows)
    fc.to_csv(OUT / "forecast_6_months.csv", index=False)
    same_cols = list(fc.columns) == list(sample.columns)

    # 4) Chart with the range
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (m, d) in zip(axes, [("headcount_end", "ALL"), ("attrition_rate_monthly_pct", "ALL")]):
        s = S[(m, d)].iloc[-36:]
        f = fc[(fc.metric == m) & (fc.department_id == d)]
        fm = pd.to_datetime(f.forecast_month)
        ax.plot(s.index, s.values, label="history")
        ax.plot(fm, f.forecast_value, "r-o", label="forecast")
        ax.fill_between(fm, f.lower_80, f.upper_80, color="red", alpha=0.2, label="80% range")
        ax.set_title(f"{m} ({d})"); ax.legend(fontsize=8)
    plt.tight_layout(); fig.savefig(OUT / "figures" / "forecast_with_range.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    h = hist[hist.department_id == "ALL"].set_index("month")["headcount_end"]
    ax.plot(h.index, h.values)
    for e in events.itertuples():
        ax.axvspan(pd.Timestamp(e.start_date), pd.Timestamp(e.end_date) + pd.offsets.MonthEnd(0), color="orange", alpha=0.25)
    ax.set_title("Company headcount with workforce events shaded"); plt.tight_layout()
    fig.savefig(OUT / "figures" / "history_with_events.png", dpi=150); plt.close(fig)

    # least trusted month = widest range relative to the value
    fc["rel_width_pct"] = (fc.upper_80 - fc.lower_80) / fc.forecast_value * 100
    worst = fc[fc.metric == "headcount_end"].sort_values("rel_width_pct", ascending=False).iloc[0]
    res = {"trained_through": trained_through, "best_method_by_backtest": best, "backtest_summary": summary.round(3).to_dict("records"),
           "columns_match_sample": bool(same_cols), "rows": int(len(fc)), "least_trusted": {"department": worst.department_id, "month": worst.forecast_month[:7],
                                                                                         "range_width_pct": round(float(worst.rel_width_pct), 1)}}
    save_json(OUT / "results.json", res)
    print(summary.round(3).to_string()); print(best, same_cols, len(fc)); print(st[st.department_id == "ALL"].round(2).to_string())
    return res


if __name__ == "__main__":
    main()
