# Task 16: Forecast the future: headcount with uncertainty
Code: `src/forecast/model.py`. Run: `python scripts/run_task16.py` (needs the workbook in `data/raw/`). Output: `outputs/forecast_6_months.csv` (72 rows, same columns as `sample_forecast`), `outputs/tables/`, `outputs/figures/`.

## What I did
1. **Plotted the history** (`figures/history_with_events.png`): 81 months, company headcount 171 to 441. Trend is up; exits are higher in May and December, lower in autumn; the COVID freeze (Apr to Jun 2020) and the hiring drives are visible as steps.
2. **Baseline:** "same as last year plus trend" (value a year ago + the change over the last year). For the attrition rate: the average of the same month in the last 3 years.
3. **Better models (statsmodels):** damped-trend exponential smoothing (ETS), and **ARIMA(1,1,1) with the workforce events as inputs** (hiring freeze = 1, hiring drive = 1; other events such as policy changes are not hiring shocks and are ignored on purpose).
4. **Events are handled, not ignored.** Past freezes and drives are switched on in the history. For the future, no event is known, so they are off; `forecast(..., planned=...)` lets HR switch a planned drive on for the forecast months.
5. **Backtest:** hide the last 6 months (and the 6 before, and the 6 before that), forecast, measure the error and how often the 80% range contained the truth.
6. **Forecast 6 months ahead** (Oct 2026 to Mar 2027) for the company, 10 departments and the monthly attrition rate, with an 80% range, in the sample's columns.

## Backtest (3 origins x 6 months, averaged over all series)
| Metric | Method | Mean abs error | MAPE % | 80% range contained the truth |
|---|---|---|---|---|
| Headcount | Baseline (last year + trend) | 2.78 | 6.9 | 73% |
| Headcount | Damped-trend ETS | **2.31** | 5.7 | 69% |
| Headcount | ARIMA with events (chosen) | 2.33 | **5.6** | 73% |
| Attrition rate (%) | Baseline (3-year same month) | 0.53 | 71 | 78% |
| Attrition rate (%) | ETS (chosen) | **0.51** | 68 | 78% |

**Why ARIMA with events was chosen for headcount although ETS has a slightly lower error:** I set a rule before looking, "if the event-aware model is within 5% of the best error, use it", because it handles events and can take planned ones. The gap is 0.02 heads, well within noise.

**Strict test the task asks for (hide the last 6 known months, company total):** baseline error 5.3 heads, ETS 6.9, ARIMA 7.3, and all three ranges contained 5 of the 6 actual months. On this one test the plain baseline is the best, so **the better model is not clearly better**; that is reported, not hidden.

## Forecast, company headcount (80% range)
| Month | Forecast | Range |
|---|---|---|
| Oct 2026 | 445.6 | 440.2 to 451.0 |
| Dec 2026 | 454.6 | 443.8 to 465.3 |
| Mar 2027 | 467.5 | 449.4 to 485.6 |

## Think for yourself: which month would you trust least?
The last month, March 2027: the range widens every month because errors add up (company total range grows from 10.8 to 36.2 heads wide). Among departments the widest relative range is **D05 in March 2027 (32% of its headcount)**, because small departments move by a few people. The **monthly attrition rate** is the least trustworthy series overall: a handful of exits per month makes it noisy (backtest error about 68%).

## Honest notes
- September 2026 added 17 hires, among the biggest months in the data (the maximum is 18, in June 2022); the models treat it as real. If it was a one-off batch, headcount is over-forecast.
- Coverage of the 80% range is 69% to 73%, a little under 80%: the ranges are slightly too narrow. Treat them as "about 7 in 10", not as a guarantee.
- The sample forecast (a different method) starts lower, from a 36-month linear trend; forecasts differ by method, which is why the range is shown.
