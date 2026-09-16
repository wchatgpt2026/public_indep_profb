import numpy as np
import pandas as pd

from nflprob.features import build_pregame_features, numeric_feature_columns


def _pbp_game(game_id, offense, defense, epa_values):
    return pd.DataFrame(
        {
            "game_id": game_id,
            "posteam": offense,
            "defteam": defense,
            "epa": epa_values,
            "play": 1,
            "success": [float(v > 0) for v in epa_values],
            "qb_dropback": 1,
            "rush": 0,
            "yards_gained": 5,
            "interception": 0,
            "fumble_lost": 0,
            "sack": 0,
            "cpoe": 1.0,
        }
    )


def test_features_are_shifted_no_current_game_leakage():
    schedule = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2025, 2025],
            "week": [1, 2],
            "gameday": ["2025-09-01", "2025-09-08"],
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "home_score": [30, 10],
            "away_score": [20, 9],
            "neutral": [False, False],
        }
    )
    pbp = pd.concat(
        [
            _pbp_game("g1", "A", "B", [1.0, 1.0]),
            _pbp_game("g1", "B", "A", [-1.0, -1.0]),
            _pbp_game("g2", "A", "B", [9.0, 9.0]),
            _pbp_game("g2", "B", "A", [-9.0, -9.0]),
        ],
        ignore_index=True,
    )
    features = build_pregame_features(schedule, pbp)
    week2 = features.loc[features["game_id"] == "g2"].iloc[0]
    assert np.isclose(week2["home_pregame_off_epa_per_play"], 1.0)
    assert np.isclose(week2["away_pregame_off_epa_per_play"], -1.0)


def test_market_columns_are_not_auto_features():
    frame = pd.DataFrame(
        {
            "elo_diff": [1.0, 2.0],
            "spread_line": [-3.5, 2.5],
            "total_line": [45.5, 47.0],
            "home_moneyline": [-150, 120],
        }
    )
    assert numeric_feature_columns(frame) == ["elo_diff"]
