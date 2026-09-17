from __future__ import annotations

import pandas as pd
import pytest

import nflprob.weekly_refit_experiment as experiment


PRIMARY_METRICS = (
    "margin_mae",
    "margin_rmse",
    "total_mae",
    "total_rmse",
    "home_win_brier",
    "home_win_log_loss",
    "home_win_ece",
    "exact_score_nll",
)


def _summary(value: float) -> dict[str, float]:
    return {metric: value for metric in PRIMARY_METRICS}


def test_weekly_refit_adds_completed_prior_weeks_to_training(monkeypatch):
    games = pd.DataFrame(
        [
            {
                "game_id": "2018_01_A_B",
                "season": 2018,
                "week": 1,
                "home_score": 24.0,
                "away_score": 20.0,
                "elo_diff": 1.0,
            },
            {
                "game_id": "2018_02_C_D",
                "season": 2018,
                "week": 2,
                "home_score": 17.0,
                "away_score": 14.0,
                "elo_diff": 2.0,
            },
            {
                "game_id": "2019_01_E_F",
                "season": 2019,
                "week": 1,
                "home_score": 21.0,
                "away_score": 20.0,
                "elo_diff": 3.0,
            },
            {
                "game_id": "2019_02_G_H",
                "season": 2019,
                "week": 2,
                "home_score": 27.0,
                "away_score": 24.0,
                "elo_diff": 4.0,
            },
        ]
    )
    training_ids: list[list[str]] = []

    class FakePredictor:
        def __init__(self, random_state: int, score_max: int) -> None:
            self.random_state = random_state
            self.score_max = score_max

        def fit(self, train, feature_columns):
            assert feature_columns == ["elo_diff"]
            training_ids.append(train["game_id"].astype(str).tolist())
            return self

    def fake_score_holdout(model, test, max_distribution_games=None):
        del model, max_distribution_games
        return pd.DataFrame({"game_id": test["game_id"].astype(str).tolist()})

    monkeypatch.setattr(experiment, "NFLPredictor", FakePredictor)
    monkeypatch.setattr(experiment, "score_holdout", fake_score_holdout)
    monkeypatch.setattr(experiment, "metrics_from_predictions", lambda frame: _summary(1.0))

    report, predictions = experiment._run_weekly_refit_backtest(
        games,
        feature_columns=["elo_diff"],
        start_season=2019,
        end_season=2019,
        min_train_games=1,
        score_max=80,
        random_state=7,
    )

    assert training_ids[0] == ["2018_01_A_B", "2018_02_C_D"]
    assert training_ids[1] == ["2018_01_A_B", "2018_02_C_D", "2019_01_E_F"]
    assert predictions["refit_week"].tolist() == [1, 2]
    assert report["summary"]["refits"] == 2.0


def test_dev_weekly_refit_rejects_confirmation_seasons():
    with pytest.raises(ValueError, match="through 2021"):
        experiment.development_weekly_refit_compare(
            pd.DataFrame(),
            validation_season=2022,
        )


def test_dev_weekly_refit_gate_uses_balanced_loss_and_point_mae(monkeypatch):
    games = pd.DataFrame(
        {
            "game_id": ["x"],
            "season": [2019],
            "week": [1],
            "home_score": [24.0],
            "away_score": [20.0],
            "elo_diff": [1.0],
            "home_qb_epa": [0.4],
            "spread_line": [3.0],
        }
    )

    def fake_frozen(
        frame,
        *,
        feature_columns,
        start_season,
        end_season,
        min_train_games,
        score_max,
        random_state,
    ):
        del end_season, min_train_games, score_max, random_state
        assert "spread_line" not in frame.columns
        assert "home_qb_epa" not in feature_columns
        value = 1.0
        return {"summary": _summary(value), "by_season": [], "skipped": []}, pd.DataFrame()

    def fake_weekly(
        frame,
        *,
        feature_columns,
        start_season,
        end_season,
        min_train_games,
        score_max,
        random_state,
    ):
        del frame, feature_columns, end_season, min_train_games, score_max, random_state
        value = 0.98 if start_season < 2021 else 0.99
        report = {"summary": {**_summary(value), "refits": 17.0}, "by_season": [], "skipped": []}
        return report, pd.DataFrame({"game_id": ["x"]})

    monkeypatch.setattr(experiment, "_run_feature_backtest", fake_frozen)
    monkeypatch.setattr(experiment, "_run_weekly_refit_backtest", fake_weekly)

    report, _ = experiment.development_weekly_refit_compare(games)

    assert report["feature_count"] == 1
    assert report["development_gate"]["passes"] is True
    assert report["selection"]["balanced_improvement_pct"] > 0.0
    assert report["validation"]["balanced_improvement_pct"] > 0.0
