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


def _pace_game(game_id, offense, defense, epa_values, current_game=False):
    n = len(epa_values)
    if current_game:
        drives = list(range(1, n + 1))
        seconds = [3600 - 10 * i for i in range(n)]
        results = ["Touchdown"] * n
        no_huddle = [1] * n
    else:
        drives = [1, 1, 2, 2]
        seconds = [3600, 3570, 3400, 3370]
        results = ["Punt", "Punt", "Touchdown", "Touchdown"]
        no_huddle = [0, 1, 0, 0]
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
            "yards_gained": 6,
            "interception": 0,
            "fumble_lost": 0,
            "sack": 0,
            "cpoe": 1.0,
            "play_id": np.arange(1, n + 1),
            "fixed_drive": drives,
            "fixed_drive_result": results,
            "game_seconds_remaining": seconds,
            "no_huddle": no_huddle,
            "down": [1, 2] * (n // 2),
            "yardline_100": [80, 60, 20, 10] if n == 4 else [10] * n,
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


def test_pace_scoring_features_are_shifted_out_of_current_game():
    schedule = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2025, 2025],
            "week": [1, 2],
            "gameday": ["2025-09-01", "2025-09-08"],
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "home_score": [30, 40],
            "away_score": [20, 10],
            "neutral": [False, False],
        }
    )
    pbp = pd.concat(
        [
            _pace_game("g1", "A", "B", [0.1, 0.2, 0.3, 0.4]),
            _pace_game("g1", "B", "A", [-0.1, -0.2, -0.3, -0.4]),
            _pace_game("g2", "A", "B", [5.0] * 8, current_game=True),
            _pace_game("g2", "B", "A", [-5.0] * 8, current_game=True),
        ],
        ignore_index=True,
    )

    features = build_pregame_features(schedule, pbp)
    week2 = features.loc[features["game_id"] == "g2"].iloc[0]

    assert np.isclose(week2["home_pregame_off_plays"], 4.0)
    assert np.isclose(week2["home_pregame_off_drives"], 2.0)
    assert np.isclose(week2["home_pregame_off_plays_per_drive"], 2.0)
    assert np.isclose(week2["home_pregame_off_seconds_per_play"], 30.0)
    assert np.isclose(week2["home_pregame_off_no_huddle_rate"], 0.25)
    assert np.isclose(week2["home_pregame_off_early_down_pass_rate"], 1.0)
    assert np.isclose(week2["home_pregame_off_red_zone_epa"], 0.35)
    assert np.isclose(week2["home_pregame_off_red_zone_success_rate"], 1.0)
    assert np.isclose(week2["home_pregame_off_scoring_drive_rate"], 0.5)


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
