from __future__ import annotations

import pandas as pd
import pytest

import nflprob.injury_margin_confirmation as confirmation


def _games() -> pd.DataFrame:
    rows = []
    for season in range(2018, 2025):
        rows.append(
            {
                "game_id": f"{season}_01_A_B",
                "season": season,
                "week": 1,
                "home_score": 24.0,
                "away_score": 20.0,
                "elo_diff": 10.0,
                "home_qb_epa": 0.5,
                "spread_line": 3.0,
            }
        )
    return pd.DataFrame(rows)


def test_injury_margin_confirmation_rejects_pre_2022_window():
    with pytest.raises(ValueError, match="2022-2024"):
        confirmation.confirmation_injury_margin_compare(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            start_season=2021,
            end_season=2024,
        )


def test_injury_margin_confirmation_uses_fixed_groups_and_total_invariance(monkeypatch):
    games = _games()
    context = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "home_t24_inj_practice_burden": 1.0,
            "home_t24_inj_status_burden": 2.0,
            "home_t24_inj_qb_status_burden": 3.0,
        }
    )
    monkeypatch.setattr(
        confirmation,
        "build_t24_injury_features",
        lambda schedules, injuries, cutoff_hours: context,
    )
    monkeypatch.setattr(confirmation, "default_feature_columns", lambda frame: ["elo_diff"])

    baseline_predictions = pd.DataFrame(
        {
            "game_id": ["2022_01_A_B"],
            "predicted_total": [44.0],
        }
    )
    candidate_predictions = pd.DataFrame(
        {
            "game_id": ["2022_01_A_B"],
            "predicted_total": [44.0],
        }
    )
    baseline_summary = {
        "margin_mae": 10.0,
        "margin_rmse": 12.0,
        "total_mae": 11.0,
        "total_rmse": 13.0,
        "home_win_brier": 0.22,
        "home_win_log_loss": 0.63,
        "home_win_ece": 0.04,
        "exact_score_nll": 8.4,
    }
    candidate_summary = {
        "margin_mae": 9.8,
        "margin_rmse": 11.8,
        "total_mae": 11.0,
        "total_rmse": 13.0,
        "home_win_brier": 0.21,
        "home_win_log_loss": 0.62,
        "home_win_ece": 0.04,
        "exact_score_nll": 8.39,
    }

    calls: dict[str, list[str]] = {}

    def fake_baseline(
        frame,
        *,
        feature_columns,
        start_season,
        end_season,
        min_train_games,
        score_max,
        random_state,
    ):
        assert "spread_line" not in frame.columns
        assert "home_qb_epa" not in feature_columns
        calls["baseline"] = list(feature_columns)
        by_season = [{"season": 2022.0, **baseline_summary}]
        return {"summary": baseline_summary, "by_season": by_season, "skipped": []}, baseline_predictions

    def fake_candidate(
        frame,
        *,
        margin_feature_columns,
        total_feature_columns,
        start_season,
        end_season,
        min_train_games,
        score_max,
        random_state,
    ):
        calls["margin"] = list(margin_feature_columns)
        calls["total"] = list(total_feature_columns)
        by_season = [{"season": 2022.0, **candidate_summary}]
        return {"summary": candidate_summary, "by_season": by_season, "skipped": []}, candidate_predictions

    monkeypatch.setattr(confirmation, "_run_feature_backtest", fake_baseline)
    monkeypatch.setattr(confirmation, "_run_target_split_backtest", fake_candidate)

    report, _ = confirmation.confirmation_injury_margin_compare(
        games,
        pd.DataFrame(),
        pd.DataFrame(),
        start_season=2022,
        end_season=2024,
    )

    assert calls["baseline"] == ["elo_diff"]
    assert calls["total"] == ["elo_diff"]
    assert "home_t24_inj_practice_burden" not in calls["margin"]
    assert "home_t24_inj_status_burden" in calls["margin"]
    assert "home_t24_inj_qb_status_burden" in calls["margin"]
    assert report["fixed_injury_groups"] == ["game_status", "position_concentration"]
    assert report["accepted"] is True
    assert report["total_invariance"]["max_abs_total_difference"] == 0.0
