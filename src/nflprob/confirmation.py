from __future__ import annotations

import pandas as pd

from .backtest import _attach_comparison_metrics, _baseline_metrics, _market_metrics
from .development import (
    COMPARISON_METRICS,
    _comparison_improvements,
    _experimental_columns_by_group,
    _model_summary,
)
from .evaluation import metrics_from_predictions, score_holdout
from .features import numeric_feature_columns
from .model import NFLPredictor, default_feature_columns


CONFIRMATION_MIN_SEASON = 2022
CONFIRMATION_GROUPS = ("play_calling", "scoring_efficiency")


def _run_confirmation_backtest(
    games: pd.DataFrame,
    *,
    feature_columns: list[str],
    start_season: int,
    end_season: int | None,
    min_train_games: int,
    score_max: int,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    completed = games.loc[games["home_score"].notna() & games["away_score"].notna()].copy()
    completed["season"] = pd.to_numeric(completed["season"], errors="raise").astype(int)
    sort_columns = [
        column
        for column in ("season", "week", "game_date", "game_id")
        if column in completed.columns
    ]
    completed = completed.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    if end_season is None:
        end_season = int(completed["season"].max())
    if end_season < start_season:
        raise ValueError("end_season must be >= start_season")

    requested = [
        season
        for season in sorted(completed["season"].unique())
        if start_season <= season <= end_season
    ]
    if not requested:
        raise ValueError("no completed games exist in the requested confirmation seasons")

    season_reports: list[dict[str, float]] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, int]] = []

    for offset, season in enumerate(requested):
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if len(train) < min_train_games:
            skipped.append(
                {
                    "season": int(season),
                    "train_games": len(train),
                    "required_train_games": int(min_train_games),
                }
            )
            continue
        if test.empty:
            continue

        model = NFLPredictor(
            random_state=random_state + offset,
            score_max=score_max,
        ).fit(train, feature_columns=feature_columns)
        predictions = score_holdout(model, test, max_distribution_games=None)

        train_margin = train["home_score"].astype(float) - train["away_score"].astype(float)
        train_total = train["home_score"].astype(float) + train["away_score"].astype(float)
        non_ties = train.loc[train["home_score"] != train["away_score"]]
        baseline_home_win = (
            float((non_ties["home_score"] > non_ties["away_score"]).mean())
            if len(non_ties)
            else 0.5
        )
        predictions["baseline_margin"] = float(train_margin.mean())
        predictions["baseline_total"] = float(train_total.mean())
        predictions["baseline_home_win_probability"] = baseline_home_win
        predictions["backtest_season"] = int(season)
        predictions["train_games"] = len(train)
        prediction_frames.append(predictions)

        model_metrics = metrics_from_predictions(predictions)
        history_metrics = _baseline_metrics(predictions)
        report = _attach_comparison_metrics(model_metrics, history_metrics)
        report.update(_market_metrics(predictions))
        report["season"] = float(season)
        report["train_games"] = float(len(train))
        season_reports.append(report)

    if not prediction_frames:
        raise ValueError(
            "no confirmation seasons could be backtested; lower --min-train-games or choose "
            "later seasons"
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    summary = _attach_comparison_metrics(
        metrics_from_predictions(all_predictions),
        _baseline_metrics(all_predictions),
    )
    summary.update(_market_metrics(all_predictions))
    summary["first_test_season"] = float(all_predictions["backtest_season"].min())
    summary["last_test_season"] = float(all_predictions["backtest_season"].max())
    summary["seasons_tested"] = float(all_predictions["backtest_season"].nunique())
    return {"summary": summary, "by_season": season_reports, "skipped": skipped}, all_predictions


def confirmation_feature_compare(
    games: pd.DataFrame,
    *,
    start_season: int = CONFIRMATION_MIN_SEASON,
    end_season: int | None = None,
    min_train_games: int = 500,
    score_max: int = 80,
    random_state: int = 7,
) -> tuple[dict[str, object], pd.DataFrame]:
    """One-shot later-era confirmation of the fixed development-selected feature block."""
    if start_season < CONFIRMATION_MIN_SEASON:
        raise ValueError("confirmation is restricted to seasons 2022 and later")

    baseline_columns = default_feature_columns(games)
    all_columns = numeric_feature_columns(games, extra_exclude={"season", "week"})
    grouped = _experimental_columns_by_group(all_columns)
    missing = [group for group in CONFIRMATION_GROUPS if not grouped.get(group)]
    if missing:
        raise ValueError(
            "confirmation feature groups are missing from the dataset: " + ", ".join(missing)
        )
    added_columns = [column for group in CONFIRMATION_GROUPS for column in grouped[group]]
    candidate_columns = baseline_columns + added_columns

    baseline_report, baseline_predictions = _run_confirmation_backtest(
        games,
        feature_columns=baseline_columns,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )
    candidate_report, candidate_predictions = _run_confirmation_backtest(
        games,
        feature_columns=candidate_columns,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )

    baseline_summary = _model_summary(baseline_report)
    candidate_summary = _model_summary(candidate_report)

    by_season: list[dict[str, object]] = []
    baseline_by_season = {
        int(float(row["season"])): row for row in baseline_report["by_season"]
    }
    for candidate_row in candidate_report["by_season"]:
        season = int(float(candidate_row["season"]))
        baseline_row = baseline_by_season[season]
        baseline_metrics = {
            metric: float(baseline_row[metric])
            for metric in COMPARISON_METRICS
            if metric in baseline_row
        }
        candidate_metrics = {
            metric: float(candidate_row[metric])
            for metric in COMPARISON_METRICS
            if metric in candidate_row
        }
        by_season.append(
            {
                "season": season,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "candidate_vs_baseline_improvement_pct": _comparison_improvements(
                    candidate_metrics,
                    baseline_metrics,
                ),
            }
        )

    same_rows = len(baseline_predictions) == len(candidate_predictions)
    if same_rows and "game_id" in baseline_predictions.columns:
        same_rows = baseline_predictions["game_id"].astype(str).tolist() == candidate_predictions[
            "game_id"
        ].astype(str).tolist()

    report: dict[str, object] = {
        "confirmation_window": {
            "start_season": int(start_season),
            "end_season": (
                int(end_season)
                if end_season is not None
                else int(candidate_predictions["backtest_season"].max())
            ),
        },
        "candidate_origin": "selected on 2019-2020 and validated once on 2021",
        "candidate_groups": list(CONFIRMATION_GROUPS),
        "feature_counts": {
            "baseline": len(baseline_columns),
            "candidate": len(candidate_columns),
            "added": len(added_columns),
        },
        "same_rows": bool(same_rows),
        "baseline_summary": baseline_report["summary"],
        "candidate_summary": candidate_report["summary"],
        "candidate_vs_baseline_improvement_pct": _comparison_improvements(
            candidate_summary,
            baseline_summary,
        ),
        "by_season": by_season,
        "baseline_skipped": baseline_report["skipped"],
        "candidate_skipped": candidate_report["skipped"],
    }
    return report, candidate_predictions
