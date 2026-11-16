"""Console logic: every function takes an httpx-style client (so tests can plug in the API directly) and returns plain text or data.
Errors become friendly messages; nothing here raises into the user interface."""
from __future__ import annotations

import json
from datetime import datetime

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABEL = "AI-generated, for human review"
FRIENDLY = {401: "The console is missing its access key. Ask the administrator.", 404: "Nothing found for that input. Check the ID and try again.",
            422: "Some of the information is not in the expected format. Check the highlighted fields.", 503: "That feature is not available right now. Please try later."}


def make_client(base_url: str, api_key: str | None = None, timeout: float = 90.0) -> httpx.Client:
    # long timeout: a sleeping free host needs about a minute to wake up
    return httpx.Client(base_url=base_url.rstrip("/"), headers={"X-API-Key": api_key} if api_key else {}, timeout=timeout)


def call(client, method: str, path: str, **kw):
    """Returns (data, error_message). Never raises."""
    try:
        r = client.request(method, path, **kw)
    except httpx.TimeoutException:
        return None, "The service is waking up or busy. Please wait a minute and try again."
    except httpx.HTTPError:
        return None, "Could not reach the service. Check your connection and the API address."
    if r.status_code >= 400:
        detail = ""
        try:
            d = r.json().get("detail")
            detail = f" ({d})" if isinstance(d, str) else ""
        except Exception:
            pass
        return None, FRIENDLY.get(r.status_code, f"The service returned an error ({r.status_code}).") + detail
    return r.json(), None


def tag(text: str) -> str:
    return f"{text}\n\n_{LABEL}_"


def status(client) -> str:
    d, e = call(client, "GET", "/v1/health")
    return f"API status: {d['status']} (models loaded {d['models_loaded_times']}x)" if d else f"API status: unavailable. {e}"


def risks_summary(client) -> tuple[pd.DataFrame, str]:
    d, e = call(client, "GET", "/v1/attrition/summary")
    if e:
        return pd.DataFrame(), e
    df = pd.DataFrame(d["top_10"]).rename(columns={"risk_score_12m": "risk score", "risk_band": "band"})
    b = d["bands"]
    return df, tag(f"{d['employees']} employees scored: {b.get('High', 0)} High, {b.get('Medium', 0)} Medium, {b.get('Low', 0)} Low. Showing the 10 highest.")


def risk_for(client, employee_id: str) -> str:
    employee_id = (employee_id or "").strip().upper()
    if not employee_id:
        return "Type an employee ID such as EMP-0004."
    d, e = call(client, "GET", f"/v1/attrition/risk/{employee_id}")
    if e:
        return e
    return tag(f"**{d['employee_id']}** ({d['department_id']}): risk {d['risk_score_12m']:.0%} in 12 months, band **{d['risk_band']}**\n\n"
               f"1. {d['top_reason_1']}\n2. {d['top_reason_2']}\n3. {d['top_reason_3']}\n\n_{d['model_note']}_")


def anomalies(client, kind: str, payload_text: str) -> tuple[pd.DataFrame, str]:
    try:
        payload = json.loads(payload_text)
    except Exception:
        return pd.DataFrame(), "That is not valid JSON. Use the example button to see the expected format."
    d, e = call(client, "POST", f"/v1/anomaly/{kind}", json=payload)
    if e:
        return pd.DataFrame(), e
    rows = d["results"]
    msg = f"{d['records_scored']} records checked, {d['records_flagged']} flagged." if rows else f"{d['records_scored']} records checked, none flagged."
    return pd.DataFrame(rows), tag(msg)


def ask(client, session_id: str, user: str, message: str, history: list):
    message = (message or "").strip()
    if not message:
        return history
    headers = {"X-Demo-User": user} if user else {}
    d, e = call(client, "POST", "/v1/chat", json={"session_id": session_id, "message": message}, headers=headers)
    reply = f"{d['reply']}\n\n_{LABEL}_" if d else e
    return history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]


def rank_resumes(client, job_id: str, resumes_text: str, top: int = 5) -> tuple[pd.DataFrame, str]:
    """One resume per block, blocks separated by a line containing only ---"""
    blocks = [b.strip() for b in (resumes_text or "").split("\n---\n") if b.strip()]
    if not blocks:
        return pd.DataFrame(), "Paste at least one resume. Separate several resumes with a line containing only ---"
    body = {"job_id": job_id, "top": int(top), "resumes": [{"resume_id": f"Resume {i + 1}", "resume_text": b} for i, b in enumerate(blocks)]}
    d, e = call(client, "POST", "/v1/resumes/rank", json=body)
    if e:
        return pd.DataFrame(), e
    df = pd.DataFrame(d["ranked"])
    df["missing_required"] = df["missing_required"].map(lambda m: ", ".join(m) or "none")
    return df, tag(d["note"])


def forecast_chart(client, metric: str, dept: str):
    d, e = call(client, "GET", "/v1/forecast", params={"metric": metric, "department_id": dept})
    if e:
        return None, e
    f = pd.DataFrame(d["rows"])
    x = pd.to_datetime(f["forecast_month"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(x, f["forecast_value"], "o-", color="#1B7F8C", label="forecast")
    ax.fill_between(x, f["lower_80"], f["upper_80"], color="#1B7F8C", alpha=0.2, label="80% range")
    ax.set_title(f"{metric} ({dept})"); ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    return fig, tag(f"{d['note']}\n\nMethod: {f['method'].iloc[0]}. Trained through {f['trained_through'].iloc[0]}.")


def pay_band(client, **fields) -> str:
    d, e = call(client, "POST", "/v1/pay/recommend", json=fields)
    if e:
        return e
    return tag(f"Recommended range: **Rs {d['low']:,.0f}** to **Rs {d['high']:,.0f}** a year (middle Rs {d['mid']:,.0f}).\n\n{d['note']}")


def monitoring(client) -> tuple[pd.DataFrame, str]:
    d, e = call(client, "GET", "/v1/monitoring")
    if e:
        return pd.DataFrame(), e
    t, p = d["twin_resume_gap"], d["pay_gap"]
    txt = (f"**Overall: {d['status']}** (as of {d['generated_utc'][:16].replace('T', ' ')} UTC)\n\n"
           f"Twin resumes: average gap {t['avg_abs_gap_points']} points ({t['verdict']}). "
           f"Pay gap: gender {p['gender']['model_gap_points']}, career gap {p['career_gap']['model_gap_points']} points ({p['verdict']}).\n\n{d['next_step']}")
    return pd.DataFrame(d["drift"]), tag(txt)
