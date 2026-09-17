import pandas as pd
import pytest

from nflprob.model import (
    NFLPredictor,
    POSTGAME_CONTEXT_FEATURES,
    default_feature_columns,
    is_experimental_feature,
    is_postgame_context_feature,
)


def test_independence_guard_rejects_market_features():
    with pytest.raises(ValueError):
        NFLPredictor._assert_independent(["elo_diff", "spread_line"])


def test_independence_guard_rejects_postgame_context():
    with pytest.raises(ValueError):
        NFLPredictor._assert_independent(["elo_diff", "home_qb_epa"])
    with pytest.raises(ValueError):
        NFLPredictor._assert_independent(["elo_diff", "temperature"])


def test_pace_scoring_features_remain_experimental_by_default():
    assert is_experimental_feature("home_pregame_off_plays")
    assert is_experimental_feature("sum_def_scoring_drive_rate_allowed")
    assert not is_experimental_feature("home_pregame_off_epa_per_play")
    assert not is_experimental_feature("elo_diff")


def test_postgame_context_is_excluded_from_deployable_defaults():
    frame = pd.DataFrame(
        {
            "home_pregame_off_epa_per_play": [0.1],
            "elo_diff": [25.0],
            "home_pregame_off_plays": [63.0],
            **{column: [1.0] for column in POSTGAME_CONTEXT_FEATURES},
        }
    )
    selected = default_feature_columns(frame)

    assert "home_pregame_off_epa_per_play" in selected
    assert "elo_diff" in selected
    assert "home_pregame_off_plays" not in selected
    assert not POSTGAME_CONTEXT_FEATURES.intersection(selected)
    assert all(is_postgame_context_feature(column) for column in POSTGAME_CONTEXT_FEATURES)
