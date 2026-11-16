"""Column lists and constants. Every Task 1 decision (D1-D20, M1-M6) maps to code through these."""

RANDOM_SEED = 42
TARGET = "left_within_12m"
DEFAULT_DATA = "data/raw/02_CortexHR_AI_ML_Dataset.xlsx"
DEFAULT_SHEET = "attrition_dataset"

# D15 / D16: columns that reveal the answer or only identify a row. Never used as inputs.
LEAKAGE_COLUMNS = [
    "exit_date", "resignation_date", "exit_reason", "exit_interview_done",
    "fnf_settlement_amount", "voluntary_left_within_12m",
]
ID_COLUMNS = ["row_id", "employee_id", "source"]
# Kept only for building time-based splits, never fed to the model.
DATE_COLUMN = "observation_date"

# Numbers taken straight from the table (scaled, capped at the 1st/99th percentile, median-filled)
NUMERIC_BASE = [
    "job_level", "tenure_months", "total_experience_years", "previous_employers_count",
    "commute_distance_km", "comp_ratio", "raises_received", "last_raise_pct",
    "months_since_last_raise", "promotions_received", "months_since_last_promotion",
    "internal_transfers", "last_performance_rating", "last_engagement_score",
    "manager_changes_12m", "training_hours_12m", "attendance_rate_3m", "late_arrivals_3m",
    "avg_overtime_hours_per_month_3m", "avg_hours_worked_3m", "wfh_share_3m",
    "unpaid_leave_days_6m", "paid_leave_days_6m", "dept_attrition_rate_prior_12m",
]
# Numbers we create (see FeatureEngineer)
NUMERIC_ENGINEERED = [
    "log_pay", "pay_vs_dept_median", "tenure_years", "attendance_gap", "overtime_per_year_of_tenure",
]
FLAG_COLUMNS = ["never_promoted", "no_raise_yet", "not_reviewed_yet", "engagement_missing"]
CATEGORICAL = ["department_id", "work_location", "work_mode", "employment_type", "highest_education"]

NUMERIC_COLUMNS = NUMERIC_BASE + NUMERIC_ENGINEERED

# D6 / D9
SALARY_TYPO_RATIO = 5.0  # pay is >5x what the role's comp_ratio implies -> typo
LOCATION_FIXES = {"bangalore": "Bengaluru", "madras": "Chennai", "delhi-ncr": "Delhi NCR", "delhi ncr": "Delhi NCR"}
