import pandas as pd

from nflprob.pregame_context_audit import audit_pregame_context_frames


def _schedules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2024_01_A_B", "2025_01_C_D"],
            "season": [2024, 2025],
            "week": [1, 1],
            "game_type": ["REG", "REG"],
            "gameday": ["2024-09-08", "2025-09-07"],
            "gametime": ["13:00", "13:00"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
        }
    )


def _injuries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "team": ["A", "A", "B"],
            "gsis_id": ["QB_A", "QB_A", "WR_B"],
            "position": ["QB", "QB", "WR"],
            "report_status": ["Questionable", "Out", "Questionable"],
            "practice_status": ["Limited", "Did Not Participate", "Full"],
            "date_modified": [
                "2024-09-06T12:00:00Z",
                "2024-09-08T16:00:00Z",
                "2024-09-06T12:00:00Z",
            ],
        }
    )


def _depth_charts() -> pd.DataFrame:
    old = pd.DataFrame(
        {
            "_requested_season": [2024, 2024],
            "season": [2024, 2024],
            "club_code": ["A", "B"],
            "week": [1, 1],
            "position": ["QB", "QB"],
            "depth_team": [1, 1],
            "gsis_id": ["QB_A", "QB_B"],
        }
    )
    new = pd.DataFrame(
        {
            "_requested_season": [2025, 2025, 2025, 2025, 2025],
            "dt": [
                "2025-09-05T11:00:00Z",
                "2025-09-06T20:00:00Z",
                "2025-09-05T11:00:00Z",
                "2025-09-05T11:00:00Z",
                "2025-09-05T11:00:00Z",
            ],
            "team": ["C", "C", "D", "C", "D"],
            "pos_abb": ["QB", "QB", "QB", "RB", "WR"],
            "pos_rank": [1, 1, 1, 1, 1],
            "gsis_id": ["QB_C_EARLY", "QB_C_LATE", "QB_D", "RB_C", "WR_D"],
        }
    )
    return pd.concat([old, new], ignore_index=True, sort=False)


def test_context_audit_enforces_fixed_cutoff_and_schema_split():
    report = audit_pregame_context_frames(
        _schedules(),
        _injuries(),
        _depth_charts(),
        start_season=2024,
        end_season=2025,
        cutoff_hours=24,
        as_of="2025-12-01T00:00:00Z",
    )

    assert report["schedule"]["matured_team_games"] == 4

    injuries = report["injuries"]
    assert injuries["requested_seasons_without_upstream_data"] == [2025]
    assert injuries["date_modified_parse_rate"] == 1.0
    assert injuries["matured_team_games"] == 2
    assert injuries["team_game_any_report_rate"] == 1.0
    assert injuries["team_game_report_available_at_cutoff_rate"] == 1.0
    assert injuries["revision_player_week_groups"] == 1

    historical = report["depth_charts"]["pre_2025_weekly"]
    assert historical["timestamped"] is False
    assert historical["strict_cutoff_enforceable"] is False
    assert historical["qb1_weekly_team_game_coverage"] == 1.0

    current = report["depth_charts"]["post_2025_timestamped"]
    assert current["timestamped"] is True
    assert current["strict_cutoff_enforceable"] is True
    assert current["qb1_team_game_coverage_at_cutoff"] == 1.0
    assert current["qb1_gsis_coverage_at_cutoff"] == 1.0
    assert current["median_snapshot_staleness_hours"] > 24.0


def test_context_audit_ignores_games_before_cutoff_matures():
    report = audit_pregame_context_frames(
        _schedules(),
        _injuries(),
        _depth_charts(),
        start_season=2025,
        end_season=2025,
        cutoff_hours=24,
        as_of="2025-09-01T00:00:00Z",
    )

    assert report["schedule"]["matured_team_games"] == 0
    current = report["depth_charts"]["post_2025_timestamped"]
    assert current["matured_team_games"] == 0
