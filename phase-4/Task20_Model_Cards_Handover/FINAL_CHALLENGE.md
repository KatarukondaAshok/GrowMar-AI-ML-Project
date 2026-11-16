# Final Challenge (yours to do, on its own branch, with no step-by-step guide)
The task asks you to choose ONE improvement and build it yourself. I cannot do this for you, because the point is your own decisions. Suggestions that come straight from this project's findings:

1. **Fix the age-proxy reasons in Task 14** (best choice: real, small, and a fairness improvement). Drop total experience from the reasons, re-run, re-check the output format and the tests.
2. **Find the cause of the monitor ALERT** on "months since last raise" and "attendance rate": confirm or reject the two hypotheses in `Task18_Monitoring/README.md`, then recalibrate or document.
3. **Widen the forecast ranges** so the 80% range really holds about 80% of the time (try conformal intervals) and re-run the backtest.
4. **A better fairness fix** for pay bands (compare removing proxies with a fairness-aware fit) using Fairlearn.

## How
```bash
git checkout -b final-challenge
# build, test (pytest -q), write what you decided below
git add . && git commit -m "Final challenge: <what you changed>" && git push -u origin final-challenge
```
## Write here (in your own words)
- What I chose and why:
- What I tried first and what did not work:
- Before and after numbers:
- What I would do next:

## Which model would I NOT release, if I were the manager? (one sentence shows judgement)
Draft to edit: "I would not release the monthly attrition-rate forecast: with only a few exits a month its error is about 68%, so a manager could read precision into noise."
