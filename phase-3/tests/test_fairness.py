import pandas as pd

from src.fairness import group_gap, risk_bands


def test_risk_bands_follow_the_stated_rule_for_430_people():
    b = risk_bands(list(range(430)))
    assert b.value_counts().to_dict() == {"Low": 301, "Medium": 86, "High": 43}
    assert b.iloc[429] == "High" and b.iloc[0] == "Low"          # the highest scores are High


def test_ties_still_give_exactly_the_right_band_sizes():
    assert risk_bands([0.1] * 100).value_counts().to_dict() == {"Low": 70, "Medium": 20, "High": 10}


def test_group_gap_reports_gap_against_the_reference_group():
    d = pd.DataFrame({"v": [10, 10, 8, 8, 8], "g": ["M", "M", "F", "F", "F"]})
    t = group_gap(d, "v", "g", "M").set_index("g")
    assert t.loc["F", "gap_vs_M"] == -2 and t.loc["M", "gap_vs_M"] == 0
