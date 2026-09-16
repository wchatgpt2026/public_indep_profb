import pytest

from nflprob.model import NFLPredictor


def test_independence_guard_rejects_market_features():
    with pytest.raises(ValueError):
        NFLPredictor._assert_independent(["elo_diff", "spread_line"])
