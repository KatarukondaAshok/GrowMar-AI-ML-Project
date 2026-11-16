"""Task 6: measure the detector for each of the 8 error types (labels are used here, never for training)."""
from __future__ import annotations

import pandas as pd

# Which injected error type each rule is meant to catch (used for per-type precision of the rules).
RULE_TO_TYPE = {"DUPLICATE_PAYMENT": "DOUBLE_PAYMENT", "ZERO_PAY": "ZERO_PAY", "NEGATIVE_NET_PAY": "NEGATIVE_NET",
                "GROSS_ABOVE_NORM": "WRONG_DECIMAL", "GROSS_BELOW_EXPECTED": "STALE_SALARY",
                "OVERTIME_ABOVE_CAP": "OVERTIME_SPIKE", "PF_NOT_DEDUCTED": "MISSED_DEDUCTION",
                "ROLE_PAY_ABOVE_PEERS": "ROLE_MISMATCH"}
TYPE_ORDER = ["DOUBLE_PAYMENT", "WRONG_DECIMAL", "MISSED_DEDUCTION", "ROLE_MISMATCH", "OVERTIME_SPIKE",
              "STALE_SALARY", "NEGATIVE_NET", "ZERO_PAY"]


def overall(flag: pd.Series, truth: pd.Series) -> dict:
    tp = int((flag & truth).sum()); fp = int((flag & ~truth).sum()); fn = int((~flag & truth).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"flagged": int(flag.sum()), "caught": tp, "missed": fn, "false_alarms": fp,
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(0 if p + r == 0 else 2 * p * r / (p + r), 3)}


def per_type(scored: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Recall for model alone, rules alone and both together, plus precision of the matching rule."""
    typ = scored.index.to_series().map(labels.set_index("payroll_item_id")["anomaly_type"])
    rows = []
    for t in TYPE_ORDER:
        is_t = typ == t
        n = int(is_t.sum())
        rule_code = next(k for k, v in RULE_TO_TYPE.items() if v == t)
        rule_col = scored[f"rule_{rule_code}"]
        rows.append({
            "error_type": t, "injected": n,
            "caught_model_alone": int((scored["model_flag"] & is_t).sum()),
            "caught_rules_alone": int((scored["rule_flag"] & is_t).sum()),
            "caught_combined": int((scored["flagged"] & is_t).sum()),
            "recall_model_alone": round((scored["model_flag"] & is_t).sum() / n, 3),
            "recall_rules_alone": round((scored["rule_flag"] & is_t).sum() / n, 3),
            "recall_combined": round((scored["flagged"] & is_t).sum() / n, 3),
            "rule_that_targets_it": rule_code,
            "rule_flags": int(rule_col.sum()),
            "rule_precision": round(int((rule_col & is_t).sum()) / max(int(rule_col.sum()), 1), 3),
        })
    return pd.DataFrame(rows)
