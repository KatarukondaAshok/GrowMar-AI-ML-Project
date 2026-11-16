# Architecture of the whole project
![architecture](architecture.png)

```mermaid
flowchart LR
  W[CortexHR workbook] --> P1[Phase 1: audit, pipeline, baselines]
  P1 --> M[7 saved models]
  W --> M
  M --> API[ONE live API /v1]
  API --> C[Insights Console]
  API --> S[Smoke test]
  API --> MON[Monitor: PSI + fairness]
  MON --> API
```

How one employee travels from raw data to a reason in the console: the raw HR row (Phase 1 cleans it and removes leakage columns) -> attrition-v2 scores it (calibrated probability) -> SHAP finds the top drivers -> a lookup turns them into plain sentences -> the row is stored in the serving table -> `GET /v1/attrition/risk/EMP-0004` returns it -> the Risks tab shows the band and 3 reasons with the label "AI-generated, for human review" -> the weekly monitor checks that the inputs have not drifted.
