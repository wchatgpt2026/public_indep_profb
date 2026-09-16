import numpy as np

from nflprob.distribution import ScoreDistributionCalibrator


def test_distribution_normalizes_and_matches_means():
    rng = np.random.default_rng(3)
    n = 600
    pred_margin = rng.normal(2.0, 6.0, n)
    pred_total = rng.normal(45.0, 7.0, n)
    home = np.clip(
        np.rint((pred_total + pred_margin) / 2 + rng.normal(0, 7, n)), 0, 60
    ).astype(int)
    away = np.clip(
        np.rint((pred_total - pred_margin) / 2 + rng.normal(0, 7, n)), 0, 60
    ).astype(int)
    seasons = rng.integers(2018, 2026, n)

    cal = ScoreDistributionCalibrator(score_max=70, neighbors=250).fit(
        home, away, pred_margin, pred_total, seasons
    )
    dist = cal.predict(predicted_margin=3.25, predicted_total=46.5, season=2026)

    assert np.isclose(dist.probabilities.sum(), 1.0)
    assert np.all(dist.probabilities >= 0)
    assert abs(dist.expected_margin - 3.25) < 0.05
    assert abs(dist.expected_total - 46.5) < 0.05
    assert 0.0 < dist.exact_score(24, 20) < 1.0
