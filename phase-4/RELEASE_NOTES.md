# Release notes: CortexHR v1.0.0
Tag: `v1.0.0` (create after the final push: `git tag v1.0.0 && git push origin v1.0.0`).

## Included
- **Phase 1:** data audit (26 decisions), feature pipeline `features-1.0.0`, baseline models, attendance anomaly detection.
- **Phase 2:** payroll detector (98 of 100 injected errors), reason codes and money at risk, anomaly API, HR intent classifier, chatbot with slots, memory and permissions.
- **Phase 3:** resume ranking (top-5 precision 0.96), fairness audit (twin gap 0.056 to 0), attrition v2 with calibration, SHAP reasons for 430 employees, fair pay bands.
- **Phase 4:** 6-month forecast with ranges, **one live API** (`/v1`, 11 endpoints, 7 models loaded once), smoke test, weekly drift and fairness monitor, Insights Console (7 tabs), model cards, this release.

## Not included
- No real employee data: everything is fictional.
- No automatic decisions: every output is a support signal for a person.
- No user accounts or database; the chatbot's HR system is a mock. The API key is a single shared key.
- Attrition and forecast results are served from the latest batch (`data/`), refreshed by re-running the scripts, not live.

## Known limits
1. Attrition v2's gain over the baseline is within noise; its value is calibration and explainability.
2. The monitor currently reports **ALERT** (drift on "months since last raise" and "attendance rate" in the 2026 scoring file). The cause is not yet confirmed; see `Task18_Monitoring/README.md`.
3. Some attrition reasons mention total experience, which can act as an age proxy. Reword before wider use.
4. Anomaly results are measured on synthetic injected errors; real-world recall will be lower.
5. Resume labels look tied to skill coverage; the twin test has only 25 pairs.
6. Forecast ranges held 69% to 73% of the time, a little under the 80% promised.
7. On a free host the API sleeps after idle time and the first call takes about a minute.
8. Docker files were written but not built in the environment where this release was prepared; the smoke test was run against the real servers started with `start.sh`.

## Numbers match the live service
Run `python smoke_test.py URL`, then compare `/v1/health` model versions with the model cards in `model_cards/`.
