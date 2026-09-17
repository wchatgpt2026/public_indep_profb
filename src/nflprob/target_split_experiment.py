from __future__ import annotations

import numpy as np
import pandas as pd

from .development import (
    COMPARISON_METRICS,
    DEVELOPMENT_MAX_SEASON,
    _comparison_improvements,
    _experimental_columns_by_group,
    _model_summary,
    _run_feature_backtest,
    _strip_market_columns,
)
from .evaluation import metrics_from_predictions, score_holdout
from .features import numeric_feature_columns
from .model import default_feature_columns
from .target_split import DevelopmentTargetSplitPredictor


TARGET_SPLIT_GROUPS = ("play_calling", "scoring_efficiency")


def _run_target_split_backtest(
    games: pd.DataFrame,
    *,
    margin_feature_columns: list[str],
    total_feature_columns: list[str],
    start_season: int,
    end_season: int,
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
    requested = [
        season
        for season in sorted(completed["season"].unique())
        if start_season <= season <= end_season
    ]
    if not requested:
        raise ValueError("no completed games exist in the requested development seasons")

    by_season: list[dict[str, float]] = []
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

        model = DevelopmentTargetSplitPredictor(
            random_state=random_state + offset,
            score_max=score_max,
        ).fit(
            train,
            margin_feature_columns=margin_feature_columns,
            total_feature_columns=total_feature_columns,
        )
        predictions = score_holdout(model, test, max_distribution_games=None)
        predictions["backtest_season"] = int(season)
        predictions["train_games"] = len(train)
        prediction_frames.append(predictions)

        metrics = metrics_from_predictions(predictions)
        metrics["season"] = float(season)
        metrics["train_games"] = float(len(train))
        by_season.append(metrics)

    if not prediction_frames:
        raise ValueError(
            "no development seasons could be backtested; lower --min-train-games or choose "
            "later development seasons"
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    summary = metrics_from_predictions(all_predictions)
    summary["first_test_season"] = float(all_predictions["backtest_season"].min())
    summary["last_test_season"] = float(all_predictions["backtest_season"].max())
    summary["seasons_tested"] = float(all_predictions["backtest_season"].nunique())
    return {"summary": summary, "by_season": by_season, "skipped": skipped}, all_predictions


def _margin_invariance(
    baseline_predictions: pd.DataFrame,
    split_predictions: pd.DataFrame,
) -> dict[str, float | bool]:
    if len(baseline_predictions) != len(split_predictions):
        return {"same_rows": False, "max_abs_margin_difference": float("nan")}
    if "game_id" in baseline_predictions.columns and "game_id" in split_predictions.columns:
        same_rows = baseline_predictions["game_id"].astype(str).tolist() == split_predictions[
            "game_id"
        ].astype(str).tolist()
    else:
        same_rows = True
    difference = np.abs(
        baseline_predictions["predicted_margin"].astype(float).to_numpy()
        - split_predictions["predicted_margin"].astype(float).to_numpy()
    )
    return {
        "same_rows": bool(same_rows),
        "max_abs_margin_difference": float(np.max(difference)) if len(difference) else 0.0,
    }


def development_target_split_compare(
    games: pd.DataFrame,
    *,
    start_season: int = 2019,
    end_season: int = DEVELOPMENT_MAX_SEASON,
    min_train_games: int = 500,
    score_max: int = 80,
    random_state: int = 7,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Test total-only augmentation on development seasons through 2021.

    Margin retains the deployable pregame-only default features. Total receives the two feature
    groups selected in the prior nested development experiment: play calling and scoring
    efficiency. No sportsbook or postgame-context field is available to either model, and 2022+
    remains untouched.
    """
    if end_season > DEVELOPMENT_MAX_SEASON:
        raise ValueError("dev-target-split is locked to seasons through 2021")
    if start_season > end_season:
        raise ValueError("start_season must be <= end_season")

    candidate_games = _strip_market_columns(games)
    all_columns = numeric_feature_columns(
        candidate_games,
        extra_exclude={"season", "week"},
    )
    baseline_columns = default_feature_columns(candidate_games)
    group_columns = _experimental_columns_by_group(all_columns)
    missing_groups = [group for group in TARGET_SPLIT_GROUPS if not group_columns.get(group)]
    if missing_groups:
        raise ValueError(
            "selected target-split groups are missing from the dataset: " + ", ".join(missing_groups)
        )
    added_total_columns = [
        column
        for group in TARGET_SPLIT_GROUPS
        for column in group_columns[group]
    ]
    total_columns = baseline_columns + added_total_columns

    baseline_report, baseline_predictions = _run_feature_backtest(
        candidate_games,
        feature_columns=baseline_columns,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )
    split_report, split_predictions = _run_target_split_backtest(
        candidate_games,
        margin_feature_columns=baseline_columns,
        total_feature_columns=total_columns,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )

    baseline_summary = _model_summary(baseline_report)
    split_summary = _model_summary(split_report)
    by_season_comparison: list[dict[str, object]] = []
    baseline_by_season = {
        int(float(row["season"])): row for row in baseline_report["by_season"]
    }
    for split_row in split_report["by_season"]:
        season = int(float(split_row["season"]))
        baseline_row = baseline_by_season[season]
        baseline_metrics = {
            metric: float(baseline_row[metric])
            for metric in COMPARISON_METRICS
            if metric in baseline_row
        }
        split_metrics = {
            metric: float(split_row[metric])
            for metric in COMPARISON_METRICS
            if metric in split_row
        }
        by_season_comparison.append(
            {
                "season": season,
                "baseline": baseline_metrics,
                "target_split": split_metrics,
                "target_split_vs_baseline_improvement_pct": _comparison_improvements(
                    split_metrics,
                    baseline_metrics,
                ),
            }
        )

    report: dict[str, object] = {
        "development_window": {
            "start_season": int(start_season),
            "end_season": int(end_season),
            "reserved_confirmation_starts": DEVELOPMENT_MAX_SEASON + 1,
        },
        "baseline_policy": "deployable pregame-only defaults",
        "hypothesis": (
            "operational margin features + selected play-calling/scoring features for total only"
        ),
        "total_feature_groups": list(TARGET_SPLIT_GROUPS),
        "feature_counts": {
            "margin": len(baseline_columns),
            "total": len(total_columns),
            "total_added": len(added_total_columns),
        },
        "baseline_summary": baseline_summary,
        "target_split_summary": split_summary,
        "target_split_vs_baseline_improvement_pct": _comparison_improvements(
            split_summary,
            baseline_summary,
        ),
        "margin_invariance": _margin_invariance(
            baseline_predictions,
            split_predictions,
        ),
        "by_season": by_season_comparison,
        "baseline_skipped": baseline_report["skipped"],
        "target_split_skipped": split_report["skipped"],
    }
    return report, split_predictions
