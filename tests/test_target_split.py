import numpy as np
import pandas as pd
import pytest

from nflprob import target_split_experiment
from nflprob.target_split import DevelopmentTargetSplitPredictor


class _FirstColumnModel:
    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


def _metrics(value=1.0):
    return {
        "margin_mae": value,
        "margin_rmse": value,
        "total_mae": value,
        "total_rmse": value,
        "home_win_brier": value,
        "home_win_log_loss": value,
        "home_win_ece": value,
        "exact_score_nll": value,
    }


def test_target_split_predict_point_uses_separate_feature_matrices():
    predictor = DevelopmentTargetSplitPredictor()
    predictor.margin_feature_columns_ = ["margin_signal"]
    predictor.total_feature_columns_ = ["total_signal"]
    predictor.margin_model_ = _FirstColumnModel()
    predictor.total_model_ = _FirstColumnModel()

    point = predictor.predict_point(pd.Series({"margin_signal": 3.5, "total_signal": 47.0}))

    assert point.predicted_margin == 3.5
    assert point.predicted_total == 47.0


def test_target_split_rejects_confirmation_seasons():
    with pytest.raises(ValueError, match="locked to seasons through 2021"):
        target_split_experiment.development_target_split_compare(
            pd.DataFrame(),
            start_season=2020,
            end_season=2022,
        )


def test_target_split_strips_market_and_augments_only_total(monkeypatch):
    games = pd.DataFrame(
        {
            "season": [2018, 2019],
            "home_score": [24, 27],
            "away_score": [20, 17],
            "elo_diff": [5.0, 10.0],
            "home_pregame_off_no_huddle_rate": [0.05, 0.10],
            "home_pregame_off_red_zone_epa": [0.01, 0.03],
            "spread_line": [1.5, 2.5],
            "home_moneyline": [-120, -130],
        }
    )
    seen = {}

    def fake_baseline(frame, *, feature_columns, **kwargs):
        del kwargs
        seen["baseline_frame"] = set(frame.columns)
        seen["baseline_features"] = list(feature_columns)
        row = {**_metrics(1.0), "season": 2019.0, "train_games": 1000.0}
        predictions = pd.DataFrame(
            {"game_id": ["g1"], "predicted_margin": [2.0]}
        )
        return {"summary": _metrics(1.0), "by_season": [row], "skipped": []}, predictions

    def fake_split(
        frame,
        *,
        margin_feature_columns,
        total_feature_columns,
        **kwargs,
    ):
        del kwargs
        seen["split_frame"] = set(frame.columns)
        seen["margin_features"] = list(margin_feature_columns)
        seen["total_features"] = list(total_feature_columns)
        row = {**_metrics(0.9), "season": 2019.0, "train_games": 1000.0}
        predictions = pd.DataFrame(
            {"game_id": ["g1"], "predicted_margin": [2.0]}
        )
        return {"summary": _metrics(0.9), "by_season": [row], "skipped": []}, predictions

    monkeypatch.setattr(target_split_experiment, "_run_feature_backtest", fake_baseline)
    monkeypatch.setattr(target_split_experiment, "_run_target_split_backtest", fake_split)

    report, _ = target_split_experiment.development_target_split_compare(
        games,
        start_season=2019,
        end_season=2019,
    )

    assert "spread_line" not in seen["baseline_frame"]
    assert "home_moneyline" not in seen["split_frame"]
    assert "home_pregame_off_no_huddle_rate" not in seen["margin_features"]
    assert "home_pregame_off_red_zone_epa" not in seen["margin_features"]
    assert "home_pregame_off_no_huddle_rate" in seen["total_features"]
    assert "home_pregame_off_red_zone_epa" in seen["total_features"]
    assert report["margin_invariance"]["max_abs_margin_difference"] == 0.0
