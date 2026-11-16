# Task 19: The Insights Console
Code: `console/app.py` (Gradio screens), `console/logic.py` (calls to the API, friendly errors). Tests: `tests/test_console.py`.

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

## Deploy
The console is a second small service. On Render add a second Web Service from `phase-4/Dockerfile.console` (port 7860) with env vars `AURA_API_URL` and `AURA_API_KEY` (set in the dashboard). Check the hosting table in `Task17_Live_API/HOSTING_DECISION.md` first: Hugging Face now lists Gradio Spaces as paid, so Render is the free option I verified.

## Still to do (only you can do these)
1. Deploy the console and write the link in the README and report.
2. **Stranger test:** hand your phone to someone who has not seen it, say nothing, and watch whether they find "Risks" first. Write down where they hesitated and fix one thing.
3. Take screenshots of 3 tabs (Risks, Anomalies, Forecast are good).

## Think for yourself: which tab would an HR manager use first?
Risks, so it is the first tab and the page opens on it. The one improvement I would make after the stranger test: a plain-language sentence at the top of each tab saying who it is for.
