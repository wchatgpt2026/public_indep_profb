import pandas as pd

from nflprob.audit import audit_dataset


def test_audit_dataset_reports_qb_weather_and_market_coverage():
    frame = pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "home_score": [20.0, 24.0, 17.0, None],
            "away_score": [17.0, 21.0, 20.0, None],
            "elo_diff": [5.0, -3.0, 2.0, 1.0],
            "home_qb_known": [1.0, 1.0, 0.0, 1.0],
            "away_qb_known": [1.0, 0.0, 0.0, 1.0],
            "home_qb_epa": [0.1, 0.0, 0.0, 0.2],
            "away_qb_epa": [0.2, 0.0, 0.0, -0.1],
            "home_qb_experience": [0.5, 0.0, 0.0, 0.25],
            "away_qb_experience": [0.75, 0.0, 0.0, 0.25],
            "temperature": [50.0, None, 70.0, 60.0],
            "wind_speed": [8.0, None, 4.0, 6.0],
            "indoors": [0.0, 1.0, 0.0, 0.0],
            "home_pregame_off_plays": [60.0, 61.0, 62.0, 63.0],
            "sum_off_plays": [120.0, 122.0, 124.0, 126.0],
            "spread_line": [3.0, None, -2.5, 1.5],
            "total_line": [45.0, 44.0, None, 47.0],
            "home_moneyline": [-150.0, None, 120.0, -110.0],
            "away_moneyline": [130.0, None, -140.0, -105.0],
        }
    )

    report = audit_dataset(frame)

    assert report["rows"] == 4
    assert report["completed_games"] == 3
    assert report["season_min"] == 2020
    assert report["season_max"] == 2021
    assert report["accepted_feature_count"] == 1
    assert report["postgame_context_policy"]["present_in_dataset"] == [
        "away_qb_epa",
        "away_qb_experience",
        "away_qb_known",
        "home_qb_epa",
        "home_qb_experience",
        "home_qb_known",
        "indoors",
        "temperature",
        "wind_speed",
    ]
    assert report["qb"]["home_known_rate"] == 0.75
    assert report["qb"]["away_known_rate"] == 0.5
    assert report["qb"]["both_known_rate"] == 0.5
    assert report["qb"]["either_unknown_rate"] == 0.5
    assert report["weather"]["temperature"]["coverage"] == 0.75
    assert report["weather"]["wind_speed"]["coverage"] == 0.75
    assert report["experimental_features"]["count"] == 2
    assert report["market_fields_evaluation_only"]["spread_line"]["coverage"] == 0.75
    assert report["market_fields_evaluation_only"]["total_line"]["coverage"] == 0.75
    assert report["market_fields_evaluation_only"]["home_moneyline"]["coverage"] == 0.75
    assert report["market_fields_evaluation_only"]["away_moneyline"]["coverage"] == 0.75
    assert report["qb"]["by_season"] == [
        {
            "season": 2020,
            "games": 2,
            "home_known_rate": 1.0,
            "away_known_rate": 0.5,
            "both_known_rate": 0.5,
        },
        {
            "season": 2021,
            "games": 2,
            "home_known_rate": 0.5,
            "away_known_rate": 0.5,
            "both_known_rate": 0.5,
        },
    ]
