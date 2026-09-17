from __future__ import annotations

import pandas as pd
import pytest

import nflprob.injury_experiment as experiment


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2021_01_A_B"],
            "season": [2021],
            "week": [1],
            "gameday": ["2021-09-12"],
            "gametime": ["13:00"],
            "home_team": ["A"],
            "away_team": ["B"],
        }
    )


def test_t24_injury_features_use_latest_eligible_revision_only():
    injuries = pd.DataFrame(
        {
            "season": [2021, 2021, 2021],
            "week": [1, 1, 1],
            "team": ["A", "A", "B"],
            "gsis_id": ["qb-a", "qb-a", "wr-b"],
            "full_name": ["A QB", "A QB", "B WR"],
            "position": ["QB", "QB", "WR"],
            "practice_status": [
                "Limited Participation",
                "Did Not Participate",
                "Did Not Participate",
            ],
            "report_status": ["Questionable", "Out", "Out"],
            "date_modified": [
                "2021-09-11T16:00:00Z",  # before the 17:00Z T-24 cutoff
                "2021-09-11T20:00:00Z",  # after cutoff and must be ignored
                "2021-09-11T16:30:00Z",
            ],
        }
    )

    features = experiment.build_t24_injury_features(_schedule(), injuries, cutoff_hours=24)
    row = features.iloc[0]

    assert row["home_t24_inj_practice_limited_count"] == 1.0
    assert row["home_t24_inj_practice_dnp_count"] == 0.0
    assert row["home_t24_inj_status_questionable_count"] == 1.0
    assert row["home_t24_inj_status_out_count"] == 0.0
    assert row["home_t24_inj_qb_practice_burden"] == 1.0
    assert row["away_t24_inj_practice_dnp_count"] == 1.0
    assert row["away_t24_inj_status_out_count"] == 1.0


def test_injury_feature_groups_are_interpretable():
    assert experiment.injury_feature_group("diff_t24_inj_practice_burden") == "practice_load"
    assert experiment.injury_feature_group("home_t24_inj_status_out_count") == "game_status"
    assert (
        experiment.injury_feature_group("sum_t24_inj_ol_practice_burden")
        == "position_concentration"
    )
    assert experiment.injury_feature_group("elo_diff") is None


def test_dev_injury_ablate_rejects_confirmation_seasons():
    with pytest.raises(ValueError, match="through 2021"):
        experiment.development_injury_group_ablation(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            validation_season=2022,
        )


def test_dev_injury_ablate_selects_then_validates_without_postgame_context(monkeypatch):
    rows = []
    for season in range(2018, 2022):
        rows.append(
            {
                "game_id": f"{season}_01_A_B",
                "season": season,
                "week": 1,
                "home_score": 24.0,
                "away_score": 20.0,
                "elo_diff": 10.0,
                "home_qb_epa": 0.4,
                "spread_line": 3.0,
            }
        )
    games = pd.DataFrame(rows)

    feature_rows = []
    for row in rows:
        feature_rows.append(
            {
                "game_id": row["game_id"],
                "home_t24_inj_practice_burden": 1.0,
                "home_t24_inj_status_burden": 2.0,
                "home_t24_inj_qb_status_burden": 3.0,
            }
        )
    monkeypatch.setattr(
        experiment,
        "build_t24_injury_features",
        lambda schedules, injuries, cutoff_hours: pd.DataFrame(feature_rows),
    )

    calls: list[dict[str, object]] = []

    def fake_backtest(
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
        assert "spread_line" not in feature_columns
        calls.append(
            {
                "features": list(feature_columns),
                "start": start_season,
                "end": end_season,
            }
        )
        has_status = any(column.endswith("t24_inj_status_burden") for column in feature_columns)
        value = 0.9 if has_status else 1.0
        summary = {
            "margin_mae": value,
            "margin_rmse": value,
            "total_mae": value,
            "total_rmse": value,
            "home_win_brier": value,
            "home_win_log_loss": value,
            "home_win_ece": value,
            "exact_score_nll": value,
        }
        predictions = pd.DataFrame({"game_id": ["x"]})
        return {"summary": summary, "by_season": [], "skipped": []}, predictions

    monkeypatch.setattr(experiment, "_run_feature_backtest", fake_backtest)

    report, _ = experiment.development_injury_group_ablation(
        games,
        pd.DataFrame(),
        pd.DataFrame(),
        selection_start_season=2019,
        selection_end_season=2020,
        validation_season=2021,
    )

    assert len(calls) == 10  # baseline + seven combinations + two validation runs
    assert report["winner"]["groups"] == ["game_status"]
    assert report["context_game_id_match_rate"] == 1.0
    assert calls[-2]["start"] == 2021
    assert calls[-1]["start"] == 2021
