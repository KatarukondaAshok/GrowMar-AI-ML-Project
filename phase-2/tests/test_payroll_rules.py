"""Task 6: every rule fires on the mistake it targets and stays quiet on a normal payslip."""
import pandas as pd

from src.payroll.rules import apply_rules, first_rule
from tests.conftest import payslip


def hits(detector, **over):
    df = pd.DataFrame([payslip(9001, emp="EMP-0001", month="2026-09-01", **over)])
    return apply_rules(df, detector.ref_).iloc[0]


def test_normal_payslip_fires_nothing(detector):
    assert not hits(detector).any()


def test_decimal_shift_is_gross_above_norm(detector):
    h = hits(detector, gross_pay=600000.0, net_pay=596000.0)
    assert h["GROSS_ABOVE_NORM"]


def test_stale_salary_is_gross_below_expected(detector):
    h = hits(detector, gross_pay=52800.0, net_pay=48800.0)
    assert h["GROSS_BELOW_EXPECTED"] and not h["GROSS_ABOVE_NORM"]


def test_negative_and_zero_net_pay(detector):
    assert hits(detector, tds=70000.0, total_deductions=72000.0, net_pay=-12000.0)["NEGATIVE_NET_PAY"]
    assert hits(detector, gross_pay=0.0, net_pay=0.0, pf_employee=0.0, professional_tax=0.0, tds=0.0, total_deductions=0.0)["ZERO_PAY"]


def test_overtime_above_cap(detector):
    assert hits(detector, overtime_hours_paid=96.0)["OVERTIME_ABOVE_CAP"]
    assert not hits(detector, overtime_hours_paid=50.0)["OVERTIME_ABOVE_CAP"]


def test_missing_pf_only_when_pf_was_paid_before(detector):
    h = hits(detector, pf_employee=0.0, total_deductions=2200.0, net_pay=57800.0)
    assert h["PF_NOT_DEDUCTED"]


def test_fixed_pay_far_above_role_peers(detector):
    assert hits(detector, fixed=170000, gross_pay=170000.0, net_pay=166000.0, monthly_fixed_gross=170000.0)["ROLE_PAY_ABOVE_PEERS"]


def test_arithmetic_mismatch(detector):
    assert hits(detector, net_pay=40000.0)["NET_PAY_MISMATCH"]


def test_duplicate_payment_flags_the_second_payslip_only(detector, history):
    first = history.iloc[[0]]
    second = first.assign(payroll_item_id="PI-D00001")
    out = apply_rules(pd.concat([first, second], ignore_index=True), detector.ref_)
    assert out["DUPLICATE_PAYMENT"].tolist() == [False, True]


def test_priority_picks_most_specific_rule(detector):
    h = pd.DataFrame([hits(detector, gross_pay=600000.0, net_pay=-5.0)])   # breaks several rules at once
    assert first_rule(h).iloc[0] == "NEGATIVE_NET_PAY"
