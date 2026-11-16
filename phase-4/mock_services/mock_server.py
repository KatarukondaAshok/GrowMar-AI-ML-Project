"""Mock HR system for Task 10 (stands in for the real HR backend; the starter kit's copy was not supplied).

Run:  uvicorn mock_services.mock_server:app --port 9000
Auth: send the header  X-Demo-User: EMP-0002  (the mock looks up that person's role, like a real login would).
Data: CSV files in MOCK_SEED_DIR, else mock_services/seed (built by scripts/build_mock_seed.py), else the small mock_services/sample_seed.
Every endpoint answers for the *calling* person only (employee=me). Payroll items are HR/ADMIN only.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query

_HERE = Path(__file__).parent
# MOCK_SEED_DIR > seed/ (built from the workbooks) > sample_seed/ (tiny demo data shipped with the repo)
SEED_DIR = Path(os.environ.get("MOCK_SEED_DIR") or (_HERE / "seed" if (_HERE / "seed" / "users.csv").exists() else _HERE / "sample_seed"))
app = FastAPI(title="Mock HR system", version="1.0.0")
_DB: dict[str, pd.DataFrame] = {}


def db() -> dict[str, pd.DataFrame]:
    """Load the CSV files once, on first use."""
    if not _DB:
        for name in ["users", "employees", "leave_balances", "leave_requests", "holidays", "payslips",
                     "attendance_monthly", "payroll_items"]:
            path = SEED_DIR / f"{name}.csv"
            _DB[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return _DB


def reload():
    _DB.clear()


def caller(x_demo_user: str | None) -> dict:
    if not x_demo_user:
        raise HTTPException(401, "Missing X-Demo-User header")
    u = db()["users"]
    row = u[u["employee_id"] == x_demo_user]
    if row.empty:
        raise HTTPException(401, "Unknown user")
    r = row.iloc[0]
    if str(r.get("is_active", "Y")) != "Y":
        raise HTTPException(403, "Account is not active")
    return {"employee_id": r["employee_id"], "role": str(r["role"]).upper()}


def employee_row(emp_id: str) -> dict:
    e = db()["employees"]
    row = e[e["employee_id"] == emp_id]
    if row.empty:
        raise HTTPException(404, "Employee not found")
    return row.iloc[0].astype(object).where(row.iloc[0].notna(), None).to_dict()


@app.get("/health")
def health():
    return {"status": "ok", "service": "mock-hr", "users": len(db()["users"])}


@app.get("/api/v1/employees/me")
def me(x_demo_user: str | None = Header(None)):
    who = caller(x_demo_user)
    e = employee_row(who["employee_id"])
    mgr = None
    if e.get("manager_id"):
        try:
            mgr = employee_row(e["manager_id"])["full_name"]
        except HTTPException:
            mgr = None
    return {"employee_id": e["employee_id"], "full_name": e["full_name"], "role": who["role"],
            "department_id": e["department_id"], "department_name": e["department_name"],
            "manager_id": e.get("manager_id"), "manager_name": mgr}


@app.get("/api/v1/employees/me/team")
def team(x_demo_user: str | None = Header(None)):
    who = caller(x_demo_user)
    e = db()["employees"]
    mine = e[e["manager_id"] == who["employee_id"]]
    return {"employee_id": who["employee_id"], "direct_reports": mine["full_name"].tolist()}


@app.get("/api/v1/employees/search")
def search(name: str, x_demo_user: str | None = Header(None)):
    """HR/ADMIN only. Several people can share a name, so the answer is a list the caller must choose from."""
    who = caller(x_demo_user)
    if who["role"] not in ("HR", "ADMIN"):
        raise HTTPException(403, "Only HR or ADMIN may search employees")
    e = db()["employees"]
    hit = e[e["full_name"].str.lower() == name.lower()]
    return {"matches": hit[["employee_id", "full_name", "department_name"]].to_dict("records")}


@app.get("/api/v1/leave/balance")
def leave_balance(employee: str = "me", x_demo_user: str | None = Header(None)):
    who = caller(x_demo_user)
    _only_me(employee)
    b = db()["leave_balances"]
    row = b[b["employee_id"] == who["employee_id"]]
    if row.empty:
        raise HTTPException(404, "No leave balance found")
    r = row.iloc[0]
    return {"employee_id": who["employee_id"], "cl": float(r["cl_balance"]), "sl": float(r["sl_balance"]), "el": float(r["el_balance"])}


@app.get("/api/v1/leave/requests")
def leave_requests(employee: str = "me", limit: int = Query(1, ge=1, le=20), x_demo_user: str | None = Header(None)):
    who = caller(x_demo_user)
    _only_me(employee)
    r = db()["leave_requests"]
    mine = r[r["employee_id"] == who["employee_id"]].sort_values("start_date", ascending=False).head(limit)
    return {"requests": [{"dates": f"{x.start_date} to {x.end_date}", "days": float(x.days), "status": str(x.status).lower()}
                         for x in mine.itertuples()]}


@app.get("/api/v1/payslips/me/{month}")
def payslip(month: str, x_demo_user: str | None = Header(None)):
    who = caller(x_demo_user)
    p = db()["payslips"]
    row = p[(p["employee_id"] == who["employee_id"]) & (p["month"] == month)]
    return {"month": month, "status": row.iloc[0]["status"] if not row.empty else "not generated yet"}


@app.get("/api/v1/attendance/summary")
def attendance_summary(month: str, employee: str = "me", x_demo_user: str | None = Header(None)):
    who = caller(x_demo_user)
    _only_me(employee)
    row = _attendance_row(who["employee_id"], month)
    return {"month": month, "present": int(row["present_days"]), "scheduled": int(row["scheduled_days"]),
            "late": int(row["late_arrivals"])}


@app.get("/api/v1/attendance/records")
def attendance_records(month: str, employee: str = "me", x_demo_user: str | None = Header(None)):
    """The raw monthly feature row, so the AI service can score it."""
    who = caller(x_demo_user)
    _only_me(employee)
    row = _attendance_row(who["employee_id"], month)
    return {"record": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}}


@app.get("/api/v1/holidays")
def holidays(after: str | None = None):
    h = db()["holidays"]
    if after:
        h = h[h["holiday_date"] >= after]
    h = h.sort_values("holiday_date")
    return {"holidays": h.head(5).to_dict("records")}


@app.get("/api/v1/payroll/items")
def payroll_items(month: str, x_demo_user: str | None = Header(None)):
    who = caller(x_demo_user)
    if who["role"] not in ("HR", "ADMIN"):
        raise HTTPException(403, "Payroll items are for HR or ADMIN only")
    p = db()["payroll_items"]
    sel = p[p["month"].astype(str).str.startswith(month)]
    return {"month": month, "count": int(len(sel)), "items": sel.astype(object).where(sel.notna(), None).to_dict("records")}


def _only_me(employee: str):
    if employee != "me":
        raise HTTPException(403, "You may only read your own data (employee=me)")


def _attendance_row(emp: str, month: str):
    a = db()["attendance_monthly"]
    row = a[(a["employee_id"] == emp) & (a["month"].astype(str).str.startswith(month))]
    if row.empty:
        raise HTTPException(404, "No attendance record for that month")
    return row.iloc[0]
