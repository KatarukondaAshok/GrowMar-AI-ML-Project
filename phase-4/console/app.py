"""Task 19: the Insights Console (Gradio).   AURA_API_URL=https://your-api AURA_API_KEY=... python console/app.py
The API address and key come from environment variables only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from console import logic

API_URL = os.environ.get("AURA_API_URL", "http://127.0.0.1:8000")
client = logic.make_client(API_URL, os.environ.get("AURA_API_KEY"))
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
JOBS = ["J01", "J02", "J03", "J04", "J05", "J06", "J07", "J08", "J09", "J10"]
EXAMPLE_RESUME = ("SUMMARY\nMachine learning engineer with 5 years of experience.\nSKILLS\nPython, SQL, Machine Learning, scikit-learn, Model Deployment\n"
                  "EXPERIENCE\nML Engineer, Acme (5.0 yrs)\n")
DEPTS = ["ALL"] + [f"D{i:02d}" for i in range(1, 11)]

with gr.Blocks(title="CortexHR Insights Console") as demo:
    gr.Markdown("# CortexHR Insights Console\nEvery result is **AI-generated, for human review**. It supports a person; it never decides.")
    status = gr.Markdown(logic.status(client))
    with gr.Tab("Risks"):
        gr.Markdown("Who may need a conversation? Reasons show what the model weighed, not what a person will do.")
        top = gr.Dataframe(label="10 highest risk scores", interactive=False)
        top_note = gr.Markdown()
        gr.Button("Show the 10 highest").click(lambda: logic.risks_summary(client), outputs=[top, top_note])
        emp = gr.Textbox(label="Employee ID", placeholder="EMP-0004")
        out = gr.Markdown()
        gr.Button("Explain this person's risk").click(lambda e: logic.risk_for(client, e), emp, out)
    with gr.Tab("Anomalies"):
        gr.Markdown("Paste payslips or attendance months as JSON, or load an example.")
        kind = gr.Radio(["payroll", "attendance"], value="payroll", label="What are you checking?")
        box = gr.Code(language="json", lines=12, label="Records")
        gr.Button("Load example").click(lambda k: (FIX / f"{k}_request.json").read_text(), kind, box)
        res = gr.Dataframe(label="Flagged records (reason and money at risk)", interactive=False)
        note = gr.Markdown()
        gr.Button("Check for mistakes").click(lambda k, p: logic.anomalies(client, k, p), [kind, box], [res, note])
    with gr.Tab("Assistant"):
        gr.Markdown("Ask about leave, payslips, attendance or policies. The assistant only answers for the person you choose below, and never shares someone else's salary.")
        user = gr.Dropdown(["EMP-0002", "EMP-0003", "EMP-0011"], value="EMP-0002", label="Who is asking? (EMP-0011 is HR)")
        chat = gr.Chatbot(height=300)
        msg = gr.Textbox(label="Your question", placeholder="How many casual leaves do I have left?")

        def send(m, h, u):
            return "", logic.ask(client, f"console-{u}", u, m, h or [])
        msg.submit(send, [msg, chat, user], [msg, chat])
        gr.Button("Send").click(send, [msg, chat, user], [msg, chat])
    with gr.Tab("Resume screener"):
        gr.Markdown("Ranks resumes for a job using skills, wording and experience. Names, gender, age, city and college are never used. Separate resumes with a line containing only `---`.")
        job = gr.Dropdown(JOBS, value="J04", label="Job")
        rtxt = gr.Textbox(lines=10, label="Resumes", value=EXAMPLE_RESUME)
        rtab = gr.Dataframe(interactive=False)
        rnote = gr.Markdown()
        gr.Button("Rank").click(lambda j, t: logic.rank_resumes(client, j, t), [job, rtxt], [rtab, rnote])
    with gr.Tab("Forecast"):
        gr.Markdown("Six months ahead. The shaded band is an 80% range: the truth should land inside it about 4 times in 5.")
        metric = gr.Dropdown(["headcount_end", "attrition_rate_monthly_pct"], value="headcount_end", label="What to forecast")
        dept = gr.Dropdown(DEPTS, value="ALL", label="Department (ALL = whole company)")
        plot = gr.Plot()
        fnote = gr.Markdown()
        gr.Button("Show forecast").click(lambda m, d: logic.forecast_chart(client, m, d), [metric, dept], [plot, fnote])
    with gr.Tab("Pay bands"):
        gr.Markdown("A fair range for a new offer. Gender and career gaps are never used.")
        with gr.Row():
            role = gr.Textbox(value="R005", label="Role ID"); dep = gr.Textbox(value="D01", label="Department ID"); lvl = gr.Slider(1, 7, value=3, step=1, label="Job level")
        with gr.Row():
            bmin = gr.Number(value=900000, label="Band minimum"); bmid = gr.Number(value=1100000, label="Band midpoint"); bmax = gr.Number(value=1400000, label="Band maximum")
        with gr.Row():
            exp = gr.Number(value=6, label="Years of experience"); sk = gr.Slider(0, 20, value=6, step=1, label="Skills count")
            loc = gr.Textbox(value="Chennai", label="Location"); edu = gr.Textbox(value="MCA", label="Highest education")
        pout = gr.Markdown()
        gr.Button("Recommend a range").click(lambda *a: logic.pay_band(client, role_id=a[0], department_id=a[1], job_level=int(a[2]), band_min_2026=a[3], band_mid_2026=a[4],
                                                                       band_max_2026=a[5], prior_experience_years=a[6], skills_count=int(a[7]), work_location=a[8], highest_education=a[9]),
                                             [role, dep, lvl, bmin, bmid, bmax, exp, sk, loc, edu], pout)
    with gr.Tab("Model health"):
        gr.Markdown("Are the models still behaving? Drift (PSI) and fairness checks, re-run every week.")
        mtab = gr.Dataframe(interactive=False)
        mnote = gr.Markdown()
        gr.Button("Show latest report").click(lambda: logic.monitoring(client), outputs=[mtab, mnote])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
