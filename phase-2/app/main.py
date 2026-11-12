"""Tasks 8 to 10: one FastAPI service with numbers (anomaly detectors) and language (HR chatbot).

Run:  uvicorn app.main:app --port 8000        (start the mock HR system first: uvicorn mock_services.mock_server:app --port 9000)
Models are loaded once at start-up (lifespan), not on every request.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date

import httpx
from fastapi import FastAPI, Header, HTTPException

from app.schemas import (AnomalyResponse, AttendanceRequest, ChatRequest, ChatResponse, PayrollRequest)
from src.chat.bot import ChatEngine
from src.chat.hrclient import HRClient
from src.chat.nlu import IntentClassifier, VERSION as CHAT_VERSION
from src.chat.resources import load_resources
from src.service import AnomalyService

MODELS_DIR = os.environ.get("AURA_MODELS_DIR", "models")
HR_URL = os.environ.get("HR_SERVICE_URL", "http://127.0.0.1:9000")
DEMO_TODAY = date.fromisoformat(os.environ.get("AURA_DEMO_TODAY", "2026-09-30"))   # the HR data runs to Sept 2026


def create_app(models_dir: str = MODELS_DIR, hr_http=None, today: date = DEMO_TODAY) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.load_count = 0
        app.state.service = AnomalyService.load(models_dir)
        clf = IntentClassifier.load(f"{models_dir}/{CHAT_VERSION}.joblib")
        res = load_resources(f"{models_dir}/chat_resources.json")
        http = hr_http or httpx.Client(base_url=HR_URL, timeout=5.0)
        app.state.engine = ChatEngine(clf, res, HRClient(http), anomaly=app.state.service, today=today)
        app.state.load_count += 1
        yield
        if hr_http is None:
            http.close()

    app = FastAPI(title="Aura Workforce AI service", version="2.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        s = app.state.service
        return {"status": "ok", "models": {**s.versions, "intent": CHAT_VERSION}, "models_loaded_times": app.state.load_count}

    @app.post("/v1/anomaly/payroll", response_model=AnomalyResponse)
    def anomaly_payroll(body: PayrollRequest):
        recs = [r.model_dump() for r in body.records]
        results = app.state.service.payroll(recs)
        return AnomalyResponse(model_version=app.state.service.versions["payroll"], records_scored=len(recs),
                               records_flagged=len(results), results=results)

    @app.post("/v1/anomaly/attendance", response_model=AnomalyResponse)
    def anomaly_attendance(body: AttendanceRequest):
        recs = [r.model_dump() for r in body.records]
        results = app.state.service.attendance(recs)
        return AnomalyResponse(model_version=app.state.service.versions["attendance"], records_scored=len(recs),
                               records_flagged=len(results), results=results)

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, x_demo_user: str | None = Header(None)):
        if x_demo_user is not None and not (len(x_demo_user) == 8 and x_demo_user.startswith("EMP-")):
            raise HTTPException(422, "X-Demo-User must look like EMP-0002")
        out = app.state.engine.reply(body.session_id, body.message, x_demo_user)
        return ChatResponse(session_id=body.session_id, model_version=CHAT_VERSION,
                            **{k: out[k] for k in ("intent", "confidence", "reply", "slots", "clarify", "refused", "source")})

    return app


app = create_app()
