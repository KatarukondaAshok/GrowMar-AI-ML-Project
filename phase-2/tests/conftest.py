"""Shared fixtures. Rule and format tests are self-contained; API and chat tests use the saved models in models/."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
needs_models = pytest.mark.skipif(not (MODELS / "payroll-anomaly-1.0.0.joblib").exists(),
                                  reason="run: python scripts/train_phase2_models.py")


def payslip(i: int, emp: str = "EMP-0001", month: str = "2026-03-01", fixed: float = 60000, role: str = "R010", **over) -> dict:
    """One internally consistent payslip (gross = fixed pay for days paid, deductions add up, net = gross - deductions)."""
    d = {"payroll_item_id": f"PI-{i:06d}", "run_id": "PR", "employee_id": emp, "month": month, "department_id": "D01",
         "role_id": role, "job_level": 2, "overtime_eligible": "N", "monthly_fixed_gross": fixed, "payable_days": 31,
         "lop_days": 0, "overtime_hours_paid": 0.0, "overtime_pay": 0.0, "variable_pay": 0.0, "gross_pay": fixed,
         "pf_employee": 1800.0, "esi_employee": 0.0, "professional_tax": 200.0, "tds": 2000.0, "other_deductions": 0.0}
    d["total_deductions"] = 4000.0
    d["net_pay"] = d["gross_pay"] - 4000.0
    d.update(over)
    return d


@pytest.fixture(scope="session")
def history():
    """60 normal payslips: 12 employees x 5 months, so every employee and role has a history."""
    rows, i = [], 0
    for e in range(12):
        for m in range(1, 6):
            i += 1
            rows.append(payslip(i, emp=f"EMP-{e + 1:04d}", month=f"2026-{m:02d}-01", fixed=55000 + 1000 * e, role="R010"))
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def detector(history):
    from src.payroll.detector import PayrollDetector
    return PayrollDetector(n_estimators=50).fit(history)


@pytest.fixture()
def mock_client(monkeypatch):
    import mock_services.mock_server as ms
    monkeypatch.setattr(ms, "SEED_DIR", ROOT / "tests/fixtures/mock_seed")
    ms.reload()
    yield TestClient(ms.app)
    ms.reload()


@pytest.fixture()
def api(mock_client):
    from app.main import create_app
    with TestClient(create_app(models_dir=str(MODELS), hr_http=mock_client, today=date(2026, 9, 30))) as c:
        yield c
