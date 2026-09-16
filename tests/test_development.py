import pandas as pd
import pytest

from nflprob import development


def _summary(
    *,
    margin_mae=10.0,
    total_mae=11.0,
    home_win_brier=0.22,
    exact_score_nll=8.0,
):
    return {
        "margin_mae": margin_mae,
        "margin_rmse": margin_mae + 3.0,
        "total_mae": total_mae,
        "total_rmse": total_mae + 3.0,
        "home_win_brier": home_win_brier,
        "home_win_log_loss": 0.63,
        "home_win_ece": 0.04,
        "exact_score_nll": exact_score_nll,
    }


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
        report = {"summary": _summary(), "by_season": [], "skipped": []}
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


def test_experimental_feature_groups_are_interpretable():
    assert development.experimental_feature_group("sum_off_plays") == "pace_volume"
    assert (
        development.experimental_feature_group("diff_def_early_down_pass_rate_allowed")
        == "play_calling"
    )
    assert (
        development.experimental_feature_group("home_pregame_off_scoring_drive_rate")
        == "scoring_efficiency"
    )
    assert development.experimental_feature_group("elo_diff") is None


def test_dev_ablate_rejects_confirmation_seasons():
    with pytest.raises(ValueError, match="locked to seasons through 2021"):
        development.development_group_ablation(
            pd.DataFrame(),
            validation_season=2022,
        )


def test_dev_ablate_selects_on_2019_2020_then_validates_2021(monkeypatch):
    games = pd.DataFrame(
        {
            "season": [2018, 2019, 2020, 2021],
            "home_score": [24, 27, 20, 31],
            "away_score": [20, 17, 23, 21],
            "elo_diff": [5.0, 10.0, -3.0, 7.0],
            "home_pregame_off_plays": [60.0, 61.0, 62.0, 63.0],
            "home_pregame_off_no_huddle_rate": [0.1, 0.2, 0.1, 0.2],
            "home_pregame_off_scoring_drive_rate": [0.3, 0.4, 0.3, 0.4],
            "spread_line": [1.5, 2.5, -1.0, 3.0],
            "home_moneyline": [-120, -130, 110, -140],
        }
    )
    calls = []

    def fake_backtest(
        frame,
        *,
        feature_columns,
        start_season,
        end_season,
        **kwargs,
    ):
        del kwargs
        assert "spread_line" not in frame.columns
        assert "home_moneyline" not in frame.columns
        groups = {
            development.experimental_feature_group(column)
            for column in feature_columns
            if development.experimental_feature_group(column) is not None
        }
        calls.append((start_season, end_season, frozenset(groups)))

        metrics = _summary()
        if start_season == 2019 and end_season == 2020:
            if groups == {"scoring_efficiency"}:
                metrics = _summary(
                    margin_mae=9.8,
                    total_mae=10.4,
                    home_win_brier=0.21,
                    exact_score_nll=7.8,
                )
            elif groups:
                metrics = _summary(
                    margin_mae=10.2,
                    total_mae=11.2,
                    home_win_brier=0.225,
                    exact_score_nll=8.1,
                )
        elif start_season == 2021 and end_season == 2021 and groups == {"scoring_efficiency"}:
            metrics = _summary(
                margin_mae=9.9,
                total_mae=10.7,
                home_win_brier=0.215,
                exact_score_nll=7.9,
            )

        report = {"summary": metrics, "by_season": [], "skipped": []}
        return report, frame.copy()

    monkeypatch.setattr(development, "_run_feature_backtest", fake_backtest)
    report, _ = development.development_group_ablation(games)

    assert report["winner"]["groups"] == ["scoring_efficiency"]
    assert report["winner"]["balanced_improvement_pct"] > 0.0
    assert report["validation"]["winner_vs_baseline_improvement_pct"]["total_mae"] > 0.0

    selection_calls = [call for call in calls if call[0:2] == (2019, 2020)]
    validation_calls = [call for call in calls if call[0:2] == (2021, 2021)]
    assert len(selection_calls) == 8  # baseline plus all seven non-empty group combinations
    assert validation_calls == [
        (2021, 2021, frozenset()),
        (2021, 2021, frozenset({"scoring_efficiency"})),
    ]
