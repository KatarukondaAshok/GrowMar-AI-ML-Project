# Task 19: The Insights Console
Code: `console/app.py` (Gradio screens), `console/logic.py` (calls to the API, friendly errors). Tests: `tests/test_console.py`.

**Live console:** https://katarukondaashok143-cortexhr-api.hf.space
**Live API:** https://katarukondaashok143-cortexhr-api.hf.space/v1/health (docs: `/docs`)
**Evidence:** [Task19_Evidence_Console.pdf](Task19_Evidence_Console.pdf), screenshots in [screenshots/](screenshots/)

## Run it
```bash
pip install -r requirements-console.txt
AURA_API_URL=https://YOUR-LIVE-API AURA_API_KEY=your-key python console/app.py     # opens on port 7860
```
Local test: start the API (`./start.sh`), then run the console with the default `AURA_API_URL=http://127.0.0.1:8000`. The API address and key come only from environment variables.

## Tabs
| Tab | What a person does |
|---|---|
| Risks | See the 10 highest risk scores; type an employee ID to get the band and 3 reasons |
| Anomalies | Load an example (or paste JSON) and see flagged payslips or attendance months with reason and money at risk |
| Assistant | Chat as EMP-0002, EMP-0003 or HR (EMP-0011); salary of others is refused |
| Resume screener | Paste resumes (separated by a `---` line), pick a job, see the ranking and missing skills |
| Forecast | Chart of the next 6 months with the 80% range, for the company or a department |
| Pay bands | Fill a short form and get a low to high range |
| Model health | Latest drift and fairness status |

## Trust design
- **Every output ends with "AI-generated, for human review"** (a test checks all seven tabs).
- **Errors are friendly:** wrong ID gives "Nothing found for that input"; invalid JSON gives "That is not valid JSON"; an unreachable or waking API gives a message to wait a minute. Nothing shows a stack trace.
- Plain numbers and plain words; no result is presented as a decision.

## Deploy (what was actually done)
The console and the API run together in one **Hugging Face Space** (Gradio type, free tier). The Space's `app.py` starts the API, loads the models once, and serves the console at `/` and the API under `/v1/...` (files in `phase-4/huggingface_space/`).
- On a free Hugging Face account the Docker type is locked, so a Gradio Space is used. The console talks to the API inside the same process; run separately, it reads `AURA_API_URL` and `AURA_API_KEY`.
- `ssr_mode=False` is needed, otherwise Gradio's front proxy answers `/v1/...` with a web page instead of the API.
- The Space was built on ZeroGPU hardware, which asks for one GPU function. `app.py` registers a dummy one on a hidden button. Nothing here uses a GPU.
- Not a final choice: Render (`render.yaml`, `Dockerfile.console`) is the alternative. See `Task17_Live_API/HOSTING_DECISION.md`.

## Evidence
- Smoke test against the live address: see `Task17_Live_API/`.
- Screenshots of 3 tabs (Risks, Forecast, Model health) are in `screenshots/`.

## Still to do (only you can do these)
1. **Stranger test:** hand your phone to someone who has not seen it, say nothing, and watch whether they find "Risks" first. Write down where they hesitated and fix one thing.
2. Open the console link on a phone and in a private window. Note the result in the evidence PDF.
3. Test the Assistant tab (own leave balance, then someone else's salary must be refused).

## Think for yourself: which tab would an HR manager use first?
Risks, so it is the first tab and the page opens on it. The one improvement I would make after the stranger test: a plain-language sentence at the top of each tab saying who it is for.

## Honest limits
- The free tier can be slow on the first request after a pause.
- The Assistant answers from a small demo HR dataset, not real records.
- The review label is text on the page; it cannot stop someone copying an output without it.
- Model health currently shows ALERT (real drift in two features); the Task 18 README explains it.