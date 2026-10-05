import pandas as pd

from trial_conversion_model.features import add_features


def test_zero_session_trial_gets_zero_shares_not_nan():
    row = {
        "sessions_day1": 0,
        "sessions_day2": 0,
        "sessions_day3": 0,
        "listen_sessions_3d": 0,
        "total_minutes_3d": 0.0,
    }
    out = add_features(pd.DataFrame([row]))
    assert out.loc[0, "day1_share"] == 0
    assert out.loc[0, "listen_share"] == 0
    assert out.loc[0, "avg_session_minutes"] == 0