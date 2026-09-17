from __future__ import annotations

import numpy as np
import pandas as pd

from .development import (
    COMPARISON_METRICS,
    _comparison_improvements,
    _model_summary,
    _run_feature_backtest,
    _strip_market_columns,
)
from .injury_experiment import (
    build_t24_injury_features,
    injury_feature_columns,
    injury_feature_group,
    load_injury_context_sources,
)
from .model import default_feature_columns
from .target_split_experiment import _run_target_split_backtest


CONFIRMATION_MIN_SEASON = 2022
CONFIRMATION_MAX_SEASON = 2024
CONFIRMATION_GROUPS = ("game_status", "position_concentration")
TOTAL_INVARIANCE_TOLERANCE = 1e-6


def _total_invariance(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> dict[str, float | bool]:
    if len(baseline_predictions) != len(candidate_predictions):
        return {"same_rows": False, "max_abs_total_difference": float("nan")}
    if "game_id" in baseline_predictions.columns and "game_id" in candidate_predictions.columns:
        same_rows = baseline_predictions["game_id"].astype(str).tolist() == candidate_predictions[
            "game_id"
        ].astype(str).tolist()
    else:
        same_rows = True
    difference = np.abs(
        baseline_predictions["predicted_total"].astype(float).to_numpy()
        - candidate_predictions["predicted_total"].astype(float).to_numpy()
    )
    return {
        "same_rows": bool(same_rows),
        "max_abs_total_difference": float(np.max(difference)) if len(difference) else 0.0,
    }


def confirmation_injury_margin_compare(
    games: pd.DataFrame,
    schedules: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    start_season: int = CONFIRMATION_MIN_SEASON,
    end_season: int = CONFIRMATION_MAX_SEASON,
    cutoff_hours: float = 24.0,
    min_train_games: int = 500,
    score_max: int = 80,
    random_state: int = 7,
) -> tuple[dict[str, object], pd.DataFrame]:
    """One-shot 2022-2024 confirmation of the fixed margin-only injury candidate."""
    if start_season < CONFIRMATION_MIN_SEASON:
        raise ValueError("injury margin confirmation is locked to seasons 2022-2024")
    if end_season > CONFIRMATION_MAX_SEASON:
        raise ValueError("injury margin confirmation cannot extend beyond the 2024 injury feed")
    if start_season > end_season:
        raise ValueError("start_season must be <= end_season")

    base_games = _strip_market_columns(games)
    baseline_columns = default_feature_columns(base_games)
    context = build_t24_injury_features(schedules, injuries, cutoff_hours=cutoff_hours)
    all_injury_columns = injury_feature_columns(context)
    selected_columns = [
        column
        for column in all_injury_columns
        if injury_feature_group(column) in CONFIRMATION_GROUPS
    ]
    if not selected_columns:
        raise ValueError("fixed injury confirmation groups are missing from the context data")

    augmented = base_games.merge(context, on="game_id", how="left", indicator="_injury_join")
    seasons = pd.to_numeric(augmented["season"], errors="coerce")
    relevant_rows = seasons.le(end_season)
    match_rate = float(augmented.loc[relevant_rows, "_injury_join"].eq("both").mean())
    if not np.isfinite(match_rate) or match_rate < 0.98:
        raise ValueError(f"injury context game_id match rate is too low: {match_rate:.3f}")
    augmented = augmented.drop(columns="_injury_join")
    augmented[all_injury_columns] = augmented[all_injury_columns].fillna(0.0).astype(float)

    margin_columns = baseline_columns + selected_columns
    total_columns = baseline_columns
    baseline_report, baseline_predictions = _run_feature_backtest(
        augmented,
        feature_columns=baseline_columns,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )
    candidate_report, candidate_predictions = _run_target_split_backtest(
        augmented,
        margin_feature_columns=margin_columns,
        total_feature_columns=total_columns,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )

    baseline_summary = _model_summary(baseline_report)
    candidate_summary = _model_summary(candidate_report)
    invariance = _total_invariance(baseline_predictions, candidate_predictions)
    acceptance_checks = {
        "margin_mae_improves": candidate_summary["margin_mae"] < baseline_summary["margin_mae"],
        "home_win_brier_improves": (
            candidate_summary["home_win_brier"] < baseline_summary["home_win_brier"]
        ),
        "exact_score_nll_not_worse": (
            candidate_summary["exact_score_nll"] <= baseline_summary["exact_score_nll"]
        ),
        "total_predictions_invariant": bool(
            invariance["same_rows"]
            and np.isfinite(float(invariance["max_abs_total_difference"]))
            and float(invariance["max_abs_total_difference"]) <= TOTAL_INVARIANCE_TOLERANCE
        ),
    }

    baseline_by_season = {
        int(float(row["season"])): row for row in baseline_report["by_season"]
    }
    by_season: list[dict[str, object]] = []
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

    report: dict[str, object] = {
        "confirmation_window": {
            "start_season": int(start_season),
            "end_season": int(end_season),
            "upstream_injury_endpoint": CONFIRMATION_MAX_SEASON,
        },
        "baseline_policy": "78-feature deployable pregame-only defaults",
        "candidate_hypothesis": (
            "add the preselected T-24 game-status + position-concentration injury block "
            "to margin only; leave total on the production baseline"
        ),
        "fixed_injury_groups": list(CONFIRMATION_GROUPS),
        "context_policy": {
            "cutoff_hours_before_kickoff": float(cutoff_hours),
            "record_rule": "latest player injury record with date_modified <= game cutoff",
        },
        "context_game_id_match_rate": match_rate,
        "feature_counts": {
            "baseline": len(baseline_columns),
            "margin_candidate": len(margin_columns),
            "margin_added": len(selected_columns),
            "total": len(total_columns),
        },
        "precommitted_acceptance_rule": {
            "requirements": [
                "margin_mae_improves",
                "home_win_brier_improves",
                "exact_score_nll_not_worse",
                "total_predictions_invariant",
            ],
            "total_invariance_tolerance": TOTAL_INVARIANCE_TOLERANCE,
        },
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "candidate_vs_baseline_improvement_pct": _comparison_improvements(
            candidate_summary,
            baseline_summary,
        ),
        "total_invariance": invariance,
        "acceptance_checks": acceptance_checks,
        "accepted": bool(all(acceptance_checks.values())),
        "by_season": by_season,
        "baseline_skipped": baseline_report["skipped"],
        "candidate_skipped": candidate_report["skipped"],
    }
    return report, candidate_predictions


def confirmation_injury_margin_compare_live(
    games: pd.DataFrame,
    **kwargs: object,
) -> tuple[dict[str, object], pd.DataFrame]:
    end_season = int(kwargs.get("end_season", CONFIRMATION_MAX_SEASON))
    seasons = pd.to_numeric(games["season"], errors="coerce").dropna().astype(int)
    if seasons.empty:
        raise ValueError("games data has no valid seasons")
    source_seasons = list(range(int(seasons.min()), end_season + 1))
    schedules, injuries = load_injury_context_sources(source_seasons)
    return confirmation_injury_margin_compare(
        games,
        schedules,
        injuries,
        **kwargs,
    )
