import pytest

from nflprob.model import NFLPredictor, is_experimental_feature


def test_independence_guard_rejects_market_features():
    with pytest.raises(ValueError):
        NFLPredictor._assert_independent(["elo_diff", "spread_line"])


def test_pace_scoring_features_remain_experimental_by_default():
    assert is_experimental_feature("home_pregame_off_plays")
    assert is_experimental_feature("sum_def_scoring_drive_rate_allowed")
    assert not is_experimental_feature("home_pregame_off_epa_per_play")
    assert not is_experimental_feature("elo_diff")
