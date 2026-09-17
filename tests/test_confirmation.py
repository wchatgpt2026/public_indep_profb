import pandas as pd
import pytest

from nflprob import confirmation


def _summary(value: float) -> dict[str, float]:
    return {
        "margin_mae": value,
        "margin_rmse": value + 1.0,
        "total_mae": value + 2.0,
        "total_rmse": value + 3.0,
        "home_win_brier": value / 50.0,
        "home_win_log_loss": value / 20.0,
        "home_win_ece": value / 100.0,
        "exact_score_nll": value + 4.0,
    }


def test_confirmation_rejects_development_seasons():
    with pytest.raises(ValueError, match="2022 and later"):
        confirmation.confirmation_feature_compare(pd.DataFrame(), start_season=2021)


def test_confirmation_uses_fixed_operational_candidate(monkeypatch):
    games = pd.DataFrame(
        {
            "season": [2021, 2022],
            "home_score": [24.0, 27.0],
            "away_score": [20.0, 21.0],
            "elo_diff": [5.0, 10.0],
            "home_qb_epa": [0.1, 0.2],
            "temperature": [55.0, 65.0],
            "home_pregame_off_plays": [60.0, 62.0],
            "home_pregame_off_no_huddle_rate": [0.05, 0.10],
            "home_pregame_off_red_zone_epa": [0.01, 0.03],
            "spread_line": [1.5, 2.5],
        }
    )
    seen: list[list[str]] = []

    def fake_backtest(frame, *, feature_columns, **kwargs):
        del frame, kwargs
        seen.append(list(feature_columns))
        value = 10.0 if len(seen) == 1 else 9.5
        row = {**_summary(value), "season": 2022.0, "train_games": 1000.0}
        predictions = pd.DataFrame(
            {"game_id": ["g1"], "backtest_season": [2022], "predicted_margin": [2.0]}
        )
        return {"summary": _summary(value), "by_season": [row], "skipped": []}, predictions

    monkeypatch.setattr(confirmation, "_run_confirmation_backtest", fake_backtest)
    report, _ = confirmation.confirmation_feature_compare(games, start_season=2022)

    baseline, candidate = seen
    assert baseline == ["elo_diff"]
    assert "home_qb_epa" not in candidate
    assert "temperature" not in candidate
    assert "spread_line" not in candidate
    assert "home_pregame_off_plays" not in candidate
    assert "home_pregame_off_no_huddle_rate" in candidate
    assert "home_pregame_off_red_zone_epa" in candidate
    assert report["candidate_groups"] == ["play_calling", "scoring_efficiency"]
    assert report["feature_counts"] == {"baseline": 1, "candidate": 3, "added": 2}
    assert report["same_rows"] is True
    assert report["candidate_vs_baseline_improvement_pct"]["margin_mae"] > 0.0
