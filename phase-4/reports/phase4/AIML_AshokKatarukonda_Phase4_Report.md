# Phase 4 Report: Release (Tasks 16 to 20)
**Name:** Katarukonda Ashok (GM00154) | **Track:** AI/ML Engineer | **Phase:** 4, Days 16 to 20 | **Repository:** https://github.com/KatarukondaAshok/GrowMar-AI-ML-Project | **Git tag:** `phase-4`
**Live API:** ADD AFTER DEPLOY | **Live console:** ADD AFTER DEPLOY

## Tasks and evidence
| Task | Done | Evidence |
|---|---|---|
| 16 Forecast with uncertainty | Done | 72-row forecast in the sample format; backtest tables; chart `Task16_Forecast/outputs/figures/forecast_with_range.png` |
| 17 One live API | Built and tested locally; deploy pending | Smoke test 11/11 on the real servers (`Task17_Live_API/smoke_test_output_local.txt`); live link: ADD AFTER DEPLOY |
| 18 Drift and fairness monitor | Built; schedule pending | `Task18_Monitoring/outputs/monitoring_report.txt`; GitHub Actions screenshot: ADD AFTER FIRST RUN |
| 19 Insights Console | Built and tested locally; deploy pending | 7 tabs, review label on every output (tests); screenshots of 3 tabs: ADD |
| 20 Model cards and handover | Done except demo and final challenge | 7 model cards, release notes, architecture picture |

## Proof it works
- **Tests:** 37 passed (forecast, monitor, 11 API endpoints, bad input 422, API key, console, no secrets in the repo).
- **Smoke test against the real running servers:** 11 of 11 checks passed; without the API key the protected calls return 401 as designed.
- **Load test (100 parallel calls, local):** 100 ok, 0 failed, slowest 2.9 s.
- **Forecast backtest (3 origins x 6 months):** headcount error 2.33 heads vs 2.78 for the baseline; attrition rate 0.51 vs 0.53 points; the 80% range held 69% to 73% of the time.
- **Monitor:** twin-resume gap 0.0 (OK); pay gap WATCH (1.68 points for career gap); drift ALERT on two features (months since last raise PSI 2.78, attendance rate PSI 1.47).

## What I learned (edit in your own words)
A forecast is only honest when it shows a range and has been tested on the past; here the better model was barely better than the simple baseline, and I reported that. A model that is not live is not finished, and going live raises practical questions such as memory, sleeping hosts, settings and secrets. Monitoring is part of the product: on its first run it raised a real alert about the 2026 data. Model cards force you to write down the limits.

## Problems I hit and how I solved them (edit in your own words)
- The monitor's first run showed ALERT on two features. I looked at the raw values instead of ignoring it: most people in the 2026 file are about 5 months from their last raise and attendance is capped at 1.0. I wrote down two hypotheses and a 4-step plan; the cause is not yet confirmed.
- Phases 2 and 3 had separate code folders with the same package name. I merged them into one package and checked that every saved model still loads.
- A free host sleeps and has little memory. I measured the service (about 290 MB) and chose a host from documented limits.

## What I would improve next
Confirm the cause of the drift alert and recalibrate; reword the attrition reasons that mention total experience (an age proxy); widen the forecast ranges so they hold 80% of the time; add real login for the console instead of one shared key.
