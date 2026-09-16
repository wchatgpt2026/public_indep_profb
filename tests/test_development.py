import pandas as pd
import pytest

from nflprob import development


def test_dev_compare_rejects_reserved_confirmation_seasons():
    with pytest.raises(ValueError, match="locked to seasons through 2021"):
        development.development_feature_compare(
            pd.DataFrame(),
            start_season=2020,
            end_season=2022,
        )


def test_dev_compare_strips_market_and_ablates_new_features(monkeypatch):
    games = pd.DataFrame(
        {
            "season": [2018, 2019, 2020, 2021],
            "home_score": [24, 27, 20, 31],
            "away_score": [20, 17, 23, 21],
            "elo_diff": [5.0, 10.0, -3.0, 7.0],
            "home_pregame_off_epa_per_play": [0.0, 0.1, 0.2, 0.3],
            "home_pregame_off_plays": [60.0, 61.0, 62.0, 63.0],
            "sum_off_plays": [120.0, 122.0, 124.0, 126.0],
            "spread_line": [1.5, 2.5, -1.0, 3.0],
            "total_line": [44.5, 45.5, 46.5, 47.5],
            "home_moneyline": [-120, -130, 110, -140],
            "away_moneyline": [105, 115, -125, 120],
        }
    )
    seen_frames = []
    seen_features = []

    def fake_backtest(frame, *, feature_columns, **kwargs):
        del kwargs
        seen_frames.append(set(frame.columns))
        seen_features.append(set(feature_columns))
        report = {
            "summary": {
                "margin_mae": 10.0,
                "margin_rmse": 13.0,
                "total_mae": 11.0,
                "total_rmse": 14.0,
                "home_win_brier": 0.22,
                "home_win_log_loss": 0.63,
                "home_win_ece": 0.04,
                "exact_score_nll": 8.0,
            },
            "by_season": [],
            "skipped": [],
        }
        return report, frame.copy()

    monkeypatch.setattr(development, "_run_feature_backtest", fake_backtest)
    report, _ = development.development_feature_compare(games)

    for frame_columns in seen_frames:
        assert "spread_line" not in frame_columns
        assert "total_line" not in frame_columns
        assert "home_moneyline" not in frame_columns
        assert "away_moneyline" not in frame_columns

    baseline_features, candidate_features = seen_features
    assert "home_pregame_off_plays" not in baseline_features
    assert "sum_off_plays" not in baseline_features
    assert "home_pregame_off_plays" in candidate_features
    assert "sum_off_plays" in candidate_features
    assert report["development_window"]["reserved_confirmation_starts"] == 2022
    assert report["feature_counts"]["added"] == 2
