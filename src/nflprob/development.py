from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import walk_forward_backtest
from .features import PACE_SCORING_METRICS, numeric_feature_columns


DEVELOPMENT_MAX_SEASON = 2021
MARKET_COLUMNS = (
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
)
COMPARISON_METRICS = (
    "margin_mae",
    "margin_rmse",
    "total_mae",
    "total_rmse",
    "home_win_brier",
    "home_win_log_loss",
    "home_win_ece",
    "exact_score_nll",
)


def pace_scoring_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return engineered feature columns added by the pace/scoring experiment."""
    selected = numeric_feature_columns(frame, extra_exclude={"season", "week"})
    return [
        column
        for column in selected
        if any(column.endswith(metric) for metric in PACE_SCORING_METRICS)
    ]


def _model_summary(report: dict[str, object]) -> dict[str, float]:
    summary = report["summary"]
    if not isinstance(summary, dict):
        raise TypeError("backtest summary must be a dictionary")
    return {
        metric: float(summary[metric])
        for metric in COMPARISON_METRICS
        if metric in summary
    }


def _improvement_pct(candidate: float, baseline: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(baseline) or baseline == 0.0:
        return float("nan")
    return float((baseline - candidate) / baseline * 100.0)


def development_feature_compare(
    games: pd.DataFrame,
    *,
    start_season: int = 2019,
    end_season: int = DEVELOPMENT_MAX_SEASON,
    min_train_games: int = 500,
    score_max: int = 80,
    random_state: int = 7,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare legacy vs pace/scoring features on the reserved development era only.

    The function intentionally refuses seasons after 2021. Market columns are removed before
    either model is fit or evaluated, so feature selection cannot be driven by sportsbook
    benchmarks. The accepted 2022+ benchmark remains a later confirmation set.
    """
    if end_season > DEVELOPMENT_MAX_SEASON:
        raise ValueError(
            "dev-compare is locked to seasons through 2021; use the regular backtest only "
            "after a development candidate has been selected"
        )
    if start_season > end_season:
        raise ValueError("start_season must be <= end_season")

    candidate_games = games.drop(columns=list(MARKET_COLUMNS), errors="ignore").copy()
    added_features = pace_scoring_feature_columns(candidate_games)
    if not added_features:
        raise ValueError(
            "no pace/scoring feature columns were found; rebuild the dataset with the current "
            "`nflprob build-data` command before running dev-compare"
        )

    baseline_games = candidate_games.drop(columns=added_features, errors="ignore")
    baseline_columns = numeric_feature_columns(
        baseline_games,
        extra_exclude={"season", "week"},
    )
    candidate_columns = numeric_feature_columns(
        candidate_games,
        extra_exclude={"season", "week"},
    )

    baseline_report, _ = walk_forward_backtest(
        baseline_games,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )
    candidate_report, candidate_predictions = walk_forward_backtest(
        candidate_games,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )

    baseline_summary = _model_summary(baseline_report)
    candidate_summary = _model_summary(candidate_report)
    improvement = {
        metric: _improvement_pct(candidate_summary[metric], baseline_summary[metric])
        for metric in COMPARISON_METRICS
        if metric in baseline_summary and metric in candidate_summary
    }

    report: dict[str, object] = {
        "development_window": {
            "start_season": int(start_season),
            "end_season": int(end_season),
            "reserved_confirmation_starts": DEVELOPMENT_MAX_SEASON + 1,
        },
        "feature_counts": {
            "baseline": len(baseline_columns),
            "candidate": len(candidate_columns),
            "added": len(added_features),
        },
        "added_features": added_features,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "candidate_vs_baseline_improvement_pct": improvement,
        "baseline_by_season": baseline_report["by_season"],
        "candidate_by_season": candidate_report["by_season"],
        "baseline_skipped": baseline_report["skipped"],
        "candidate_skipped": candidate_report["skipped"],
    }
    return report, candidate_predictions
