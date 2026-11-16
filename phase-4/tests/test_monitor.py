import numpy as np
import pandas as pd
import pytest

from src.monitor.monitor import PSI_ALERT, PSI_WATCH, drift_table, format_report, psi, run, verdict


def test_psi_zero_for_same_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < PSI_WATCH
    assert psi(a, rng.normal(0.5, 1, 5000)) > PSI_WATCH
    assert psi(a, rng.normal(3, 1, 5000)) > PSI_ALERT


def test_psi_handles_constant_and_missing():
    assert psi([1, 1, 1, 1], [1, 1, 1]) == 0.0
    assert psi(pd.Series([1, 2, 3, np.nan] * 50), pd.Series([1, 2, 3] * 50)) >= 0


@pytest.mark.parametrize("v,exp", [(0.05, "OK"), (0.10, "OK"), (0.11, "WATCH"), (0.25, "WATCH"), (0.26, "ALERT")])
def test_verdict_boundaries(v, exp):
    assert verdict(v, PSI_WATCH, PSI_ALERT) == exp


def test_drift_table_has_8_features_with_verdicts():
    ref = pd.read_csv("data/monitor/psi_reference.csv")
    t = drift_table(ref, ref.sample(400, random_state=1))
    assert len(t) == 8 and set(t.verdict) <= {"OK", "WATCH", "ALERT"} and (t.verdict == "OK").all()


def test_full_monitor_run_and_report(tmp_path):
    import shutil
    shutil.copytree("data/monitor", tmp_path / "monitor")
    rep = run(data_dir=tmp_path, write=True)
    assert rep["status"] in ("OK", "WATCH", "ALERT") and (tmp_path / "monitoring_latest.json").exists()
    assert rep["twin_resume_gap"]["pairs"] == 25 and rep["twin_resume_gap"]["verdict"] == "OK"     # masked screener: twins score the same
    assert "OVERALL" in format_report(rep)
