"""Build the mock HR system's CSV seed from the two workbooks (run once; the seed is not committed)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORTEX = ROOT / "data/raw/02_CortexHR_AI_ML_Dataset.xlsx"
LEDGER = ROOT / "data/raw/01_LedgerForge_Python_Master_Dataset.xlsx"
OUT = ROOT / "mock_services/seed"


def main():
    for f in (CORTEX, LEDGER):
        if not f.exists():
            sys.exit(f"Missing {f}. Put the workbook there (see data/raw/README.md).")
    OUT.mkdir(parents=True, exist_ok=True)
    L = pd.read_excel(LEDGER, sheet_name=["users", "employees", "departments", "leave_balances", "leave_requests", "holidays"])
    C = pd.read_excel(CORTEX, sheet_name=["payroll_monthly_features", "attendance_monthly_features"])
    L["users"][["employee_id", "username", "role", "is_active"]].to_csv(OUT / "users.csv", index=False)
    emp = L["employees"].merge(L["departments"][["department_id", "department_name"]], on="department_id", how="left")
    emp[["employee_id", "full_name", "manager_id", "department_id", "department_name"]].to_csv(OUT / "employees.csv", index=False)
    bal = L["leave_balances"].sort_values("as_of_date").drop_duplicates("employee_id", keep="last")
    bal[["employee_id", "cl_balance", "sl_balance", "el_balance"]].to_csv(OUT / "leave_balances.csv", index=False)
    lr = L["leave_requests"].copy()
    for c in ("start_date", "end_date"):
        lr[c] = pd.to_datetime(lr[c]).dt.strftime("%Y-%m-%d")
    lr[["employee_id", "start_date", "end_date", "days", "status"]].to_csv(OUT / "leave_requests.csv", index=False)
    h = L["holidays"].copy(); h["holiday_date"] = pd.to_datetime(h["holiday_date"]).dt.strftime("%Y-%m-%d")
    h[["holiday_date", "holiday_name"]].to_csv(OUT / "holidays.csv", index=False)
    pay = C["payroll_monthly_features"].copy(); pay["month"] = pd.to_datetime(pay["month"]).dt.strftime("%Y-%m-%d")
    pay.to_csv(OUT / "payroll_items.csv", index=False)
    ps = pay[["employee_id", "month"]].drop_duplicates().assign(status="ready to download")
    ps["month"] = ps["month"].str[:7]
    ps.to_csv(OUT / "payslips.csv", index=False)
    att = C["attendance_monthly_features"].copy(); att["month"] = pd.to_datetime(att["month"]).dt.strftime("%Y-%m-%d")
    att.to_csv(OUT / "attendance_monthly.csv", index=False)
    print("seed written to", OUT, {p.name: len(pd.read_csv(p)) for p in sorted(OUT.glob('*.csv'))})


if __name__ == "__main__":
    main()
