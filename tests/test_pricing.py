import numpy as np

from nflprob.distribution import JointScoreDistribution
from nflprob.pricing import probability_to_american


def test_market_pricing_push_and_complements():
    probs = np.zeros((8, 8))
    probs[7, 3] = 0.40
    probs[3, 7] = 0.35
    probs[7, 7] = 0.25
    dist = JointScoreDistribution(probs)

    home = dist.moneyline("home")
    away = dist.moneyline("away")
    assert np.isclose(home.push_probability, 0.25)
    assert np.isclose(home.fair_probability + away.fair_probability, 1.0)

    spread = dist.spread(-4, "home")
    assert np.isclose(spread.push_probability, 0.40)

    total = dist.total(10, "over")
    assert np.isclose(total.push_probability, 0.75)


def test_american_odds_conversion():
    assert np.isclose(probability_to_american(0.5), 100.0)
    assert np.isclose(probability_to_american(0.6), -150.0)
    assert np.isclose(probability_to_american(0.4), 150.0)
