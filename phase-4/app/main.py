"""Task 17: ONE versioned API for every CortexHR model.  Run:  uvicorn app.main:app --port 8000

Settings come only from environment variables (no secret is stored in the repository):
  AURA_MODELS_DIR   folder with the saved models            (default: models)
  AURA_DATA_DIR     folder with the serving tables          (default: data)
  HR_SERVICE_URL    mock HR system used by the chatbot      (default: http://127.0.0.1:9000)
  AURA_API_KEY      if set, every /v1 call (except health) must send header  X-API-Key
  AURA_DEMO_TODAY   'today' for the demo chatbot            (default: 2026-09-30)
All models are loaded ONCE at start-up (lifespan), never per request.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query

from app.schemas import AnomalyResponse, AttendanceRequest, ChatRequest, ChatResponse, PayrollRequest
from app.schemas_v4 import (REVIEW_LABEL, AttritionRecords, PayRequest, PayResponse, RankedResume, RankRequest, RankResponse)
from src.attrition.v2 import AttritionModel
from src.chat.bot import ChatEngine
from src.chat.hrclient import HRClient
from src.chat.nlu import VERSION as CHAT_VERSION, IntentClassifier
from src.chat.resources import load_resources
from src.fairness import risk_bands
from src.pay.model import PayBandModel
from src.service import AnomalyService

warnings.filterwarnings("ignore")
API_VERSION = "v1"
RESUME_MODEL = "resume-screener-1.0.0"
ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def create_app(models_dir: str | None = None, data_dir: str | None = None, hr_http=None, today: date | None = None,
               api_key: str | None = None) -> FastAPI:
    models_dir = models_dir or _env("AURA_MODELS_DIR", str(ROOT / "models"))
    data_dir = data_dir or _env("AURA_DATA_DIR", str(ROOT / "data"))
    today = today or date.fromisoformat(_env("AURA_DEMO_TODAY", "2026-09-30"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.time()
        app.state.load_count = 0
        app.state.service = AnomalyService.load(models_dir)
        clf = IntentClassifier.load(f"{models_dir}/{CHAT_VERSION}.joblib")
        res = load_resources(f"{models_dir}/chat_resources.json")
        http = hr_http or httpx.Client(base_url=_env("HR_SERVICE_URL", "http://127.0.0.1:9000"), timeout=5.0)
        app.state.engine = ChatEngine(clf, res, HRClient(http), anomaly=app.state.service, today=today)
        app.state.attrition = AttritionModel.load(f"{models_dir}/attrition-v2.joblib")
        app.state.pay = PayBandModel.load(f"{models_dir}/paybands-v1.joblib")
        rp = Path(models_dir) / f"{RESUME_MODEL}.joblib"
        app.state.resume = joblib.load(rp)["model"] if rp.exists() else None
        d = Path(data_dir)
        app.state.risk = pd.read_csv(d / "attrition_risk_2026.csv") if (d / "attrition_risk_2026.csv").exists() else pd.DataFrame()
        app.state.forecast = pd.read_csv(d / "forecast_6_months.csv") if (d / "forecast_6_months.csv").exists() else pd.DataFrame()
        app.state.input_cols = json.loads((d / "attrition_input_columns.json").read_text())["required_columns"] if (d / "attrition_input_columns.json").exists() else []
        app.state.monitor_path = d / "monitoring_latest.json"
        app.state.load_count += 1
        app.state.load_seconds = round(time.time() - t0, 2)
        yield
        if hr_http is None:
            http.close()

    key = api_key if api_key is not None else os.environ.get("AURA_API_KEY")

    def auth(x_api_key: str | None = Header(None)):
        if key and x_api_key != key:
            raise HTTPException(401, "Missing or wrong X-API-Key header")

    app = FastAPI(title="CortexHR live API", version="1.0.0", lifespan=lifespan,
                  description="Anomaly detection, HR chatbot, resume ranking, attrition risk, pay bands, forecast and monitoring in one service.")
    guard = [Depends(auth)]

    @app.get("/v1/health")
    def health():
        s = app.state
        return {"status": "ok", "api_version": API_VERSION, "models_loaded_times": s.load_count, "startup_seconds": s.load_seconds,
                "models": {**s.service.versions, "intent": CHAT_VERSION, "attrition": "attrition-v2", "pay_bands": "paybands-v1",
                           "resume_screener": RESUME_MODEL if s.resume is not None else "not loaded", "forecast": "headcount-forecast-1.0.0"},
                "auth_required": bool(key)}

    @app.get("/health", include_in_schema=False)
    def health_alias():
        return health()

    # ---- anomaly (Tasks 6 to 8) ------------------------------------------------------------------------
    @app.post("/v1/anomaly/payroll", response_model=AnomalyResponse, dependencies=guard)
    def anomaly_payroll(body: PayrollRequest):
        recs = [r.model_dump() for r in body.records]
        res = app.state.service.payroll(recs)
        return AnomalyResponse(model_version=app.state.service.versions["payroll"], records_scored=len(recs), records_flagged=len(res), results=res)

    @app.post("/v1/anomaly/attendance", response_model=AnomalyResponse, dependencies=guard)
    def anomaly_attendance(body: AttendanceRequest):
        recs = [r.model_dump() for r in body.records]
        res = app.state.service.attendance(recs)
        return AnomalyResponse(model_version=app.state.service.versions["attendance"], records_scored=len(recs), records_flagged=len(res), results=res)

    # ---- assistant (Tasks 9 and 10) --------------------------------------------------------------------
    @app.post("/v1/chat", response_model=ChatResponse, dependencies=guard)
    def chat(body: ChatRequest, x_demo_user: str | None = Header(None)):
        if x_demo_user is not None and not (len(x_demo_user) == 8 and x_demo_user.startswith("EMP-")):
            raise HTTPException(422, "X-Demo-User must look like EMP-0002")
        out = app.state.engine.reply(body.session_id, body.message, x_demo_user)
        return ChatResponse(session_id=body.session_id, model_version=CHAT_VERSION,
                            **{k: out[k] for k in ("intent", "confidence", "reply", "slots", "clarify", "refused", "source")})

    # ---- resume screener (Tasks 11 and 12) -------------------------------------------------------------
    @app.post("/v1/resumes/rank", response_model=RankResponse, dependencies=guard)
    def rank(body: RankRequest):
        sc = app.state.resume
        if sc is None:
            raise HTTPException(503, "Resume screener is not loaded (run scripts/train_phase4_models.py)")
        if body.job_id not in sc.jobs_.index:
            raise HTTPException(404, f"Unknown job_id {body.job_id}. Known: {sorted(sc.jobs_.index)}")
        df = pd.DataFrame([r.model_dump() for r in body.resumes])
        df["candidate_name"], df["city"] = df["candidate_name"].fillna(""), df["city"].fillna("")
        top = sc.rank(df, body.job_id, top=body.top)
        ranked = [RankedResume(resume_id=r.resume_id, score=float(r.score), required_coverage=float(r.required_coverage),
                               missing_required=list(r.missing_required), years_found=None if pd.isna(r.years_found) else float(r.years_found))
                  for r in top.itertuples()]
        return RankResponse(job_id=body.job_id, model_version=RESUME_MODEL, ranked=ranked)

    # ---- attrition (Tasks 13 and 14) -------------------------------------------------------------------
    @app.get("/v1/attrition/risk/{employee_id}", dependencies=guard)
    def attrition_risk(employee_id: str):
        r = app.state.risk
        row = r[r.employee_id == employee_id]
        if row.empty:
            raise HTTPException(404, f"No risk row for {employee_id}")
        out = row.iloc[0].to_dict()
        out["review_label"] = REVIEW_LABEL
        return out

    @app.get("/v1/attrition/summary", dependencies=guard)
    def attrition_summary():
        r = app.state.risk
        return {"employees": int(len(r)), "bands": r.risk_band.value_counts().to_dict(), "mean_risk": round(float(r.risk_score_12m.mean()), 4),
                "top_10": r.sort_values("risk_score_12m", ascending=False).head(10)[["employee_id", "department_id", "risk_score_12m", "risk_band", "top_reason_1", "top_reason_2", "top_reason_3"]].to_dict("records"),
                "review_label": REVIEW_LABEL}

    @app.post("/v1/attrition/score", dependencies=guard)
    def attrition_score(body: AttritionRecords):
        missing = [c for c in app.state.input_cols if c != "observation_date" and any(c not in r for r in body.records)]
        if missing:
            raise HTTPException(422, f"Missing required columns: {missing[:8]}")
        df = pd.DataFrame(body.records)
        try:
            p = app.state.attrition.predict_proba(df)
        except Exception as e:
            raise HTTPException(422, f"Could not score these records: {e}")
        band = risk_bands(p) if len(p) >= 10 else None
        return {"model_version": "attrition-v2", "scores": [{"employee_id": r.get("employee_id"), "risk_score_12m": round(float(s), 3)} for r, s in zip(body.records, p)],
                "note": "Calibrated 12-month probability. Support signal, needs human review.", "review_label": REVIEW_LABEL}

    # ---- pay bands (Task 15) ---------------------------------------------------------------------------
    @app.post("/v1/pay/recommend", response_model=PayResponse, dependencies=guard)
    def pay(body: PayRequest):
        df = pd.DataFrame([body.model_dump()])
        r = app.state.pay.recommend(df, as_of_now=True).iloc[0]
        return PayResponse(model_version="paybands-v1", low=float(r.low), mid=float(r.mid), high=float(r.high))

    # ---- forecast (Task 16) ----------------------------------------------------------------------------
    @app.get("/v1/forecast", dependencies=guard)
    def forecast(metric: str = Query("headcount_end"), department_id: str = Query("ALL")):
        f = app.state.forecast
        sel = f[(f.metric == metric) & (f.department_id == department_id)]
        if sel.empty:
            raise HTTPException(404, f"No forecast for metric={metric} department_id={department_id}")
        return {"metric": metric, "department_id": department_id, "horizon_months": int(len(sel)), "rows": sel.to_dict("records"),
                "note": "A range, not a promise: the 80% range should contain the true value about 4 times in 5.", "review_label": REVIEW_LABEL}

    # ---- monitoring (Task 18) --------------------------------------------------------------------------
    @app.get("/v1/monitoring", dependencies=guard)
    def monitoring():
        p = app.state.monitor_path
        if not p.exists():
            raise HTTPException(404, "No monitoring report yet. Run: python -m src.monitor.monitor")
        return json.loads(p.read_text())

    return app


app = create_app()
