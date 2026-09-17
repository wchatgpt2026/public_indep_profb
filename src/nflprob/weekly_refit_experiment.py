from __future__ import annotations

import numpy as np
import pandas as pd

from .development import (
    COMPARISON_METRICS,
    DEVELOPMENT_MAX_SEASON,
    SELECTION_METRICS,
    _comparison_improvements,
    _model_summary,
    _normalized_selection_loss,
    _run_feature_backtest,
    _strip_market_columns,
)
from .evaluation import metrics_from_predictions, score_holdout
from .model import NFLPredictor, default_feature_columns


def _run_weekly_refit_backtest(
    games: pd.DataFrame,
    *,
    feature_columns: list[str],
    start_season: int,
    end_season: int,
    min_train_games: int,
    score_max: int,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Refit at each NFL week boundary using only completed earlier weeks.

    Week 1 uses prior seasons only. Week N uses all prior seasons plus completed games from
    weeks < N in the same season. Games earlier in the same NFL week are intentionally not used,
    which keeps the split simple and auditable.
    """
    required = {"season", "week", "home_score", "away_score"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"weekly refit data missing columns: {sorted(missing)}")

    completed = games.loc[games["home_score"].notna() & games["away_score"].notna()].copy()
    completed["season"] = pd.to_numeric(completed["season"], errors="raise").astype(int)
    completed["__week"] = pd.to_numeric(completed["week"], errors="raise").astype(int)
    sort_columns = [
        column
        for column in ("season", "__week", "game_date", "game_id")
        if column in completed.columns
    ]
    completed = completed.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    requested = [
        season
        for season in sorted(completed["season"].unique())
        if start_season <= season <= end_season
    ]
    if not requested:
        raise ValueError("no completed games exist in the requested weekly-refit seasons")

    by_season: list[dict[str, float]] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, int]] = []
    refits = 0

    for season_offset, season in enumerate(requested):
        season_frames: list[pd.DataFrame] = []
        season_games = completed.loc[completed["season"] == season]
        for week in sorted(int(value) for value in season_games["__week"].unique()):
            train = completed.loc[
                (completed["season"] < season)
                | ((completed["season"] == season) & (completed["__week"] < week))
            ].drop(columns="__week")
            test = completed.loc[
                (completed["season"] == season) & (completed["__week"] == week)
            ].drop(columns="__week")
            if len(train) < min_train_games:
                skipped.append(
                    {
                        "season": int(season),
                        "week": int(week),
                        "train_games": len(train),
                        "required_train_games": int(min_train_games),
                    }
                )
                continue
            if test.empty:
                continue

            model = NFLPredictor(
                random_state=random_state + season_offset,
                score_max=score_max,
            ).fit(train, feature_columns=feature_columns)
            predictions = score_holdout(model, test, max_distribution_games=None)
            predictions["backtest_season"] = int(season)
            predictions["refit_week"] = int(week)
            predictions["train_games"] = len(train)
            season_frames.append(predictions)
            prediction_frames.append(predictions)
            refits += 1

        if season_frames:
            season_predictions = pd.concat(season_frames, ignore_index=True, sort=False)
            metrics = metrics_from_predictions(season_predictions)
            metrics["season"] = float(season)
            metrics["refits"] = float(len(season_frames))
            metrics["last_train_games"] = float(season_frames[-1]["train_games"].iloc[0])
            by_season.append(metrics)

    if not prediction_frames:
        raise ValueError(
            "no weekly refits could be evaluated; lower --min-train-games or choose later seasons"
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    summary = metrics_from_predictions(all_predictions)
    summary["first_test_season"] = float(all_predictions["backtest_season"].min())
    summary["last_test_season"] = float(all_predictions["backtest_season"].max())
    summary["seasons_tested"] = float(all_predictions["backtest_season"].nunique())
    summary["refits"] = float(refits)
    return {"summary": summary, "by_season": by_season, "skipped": skipped}, all_predictions


def _primary_nonworse(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    return bool(
        candidate["margin_mae"] <= baseline["margin_mae"]
        and candidate["total_mae"] <= baseline["total_mae"]
    )


def development_weekly_refit_compare(
    games: pd.DataFrame,
    *,
    selection_start_season: int = 2019,
    selection_end_season: int = 2020,
    validation_season: int = 2021,
    min_train_games: int = 500,
    score_max: int = 80,
    random_state: int = 7,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare season-frozen fitting with weekly expanding refits before 2022."""
    if validation_season > DEVELOPMENT_MAX_SEASON:
        raise ValueError("dev-weekly-refit validation is locked to seasons through 2021")
    if selection_end_season >= validation_season:
        raise ValueError("selection_end_season must be earlier than validation_season")
    if selection_start_season > selection_end_season:
        raise ValueError("selection_start_season must be <= selection_end_season")

    candidate_games = _strip_market_columns(games)
    baseline_columns = default_feature_columns(candidate_games)

    frozen_selection_report, _ = _run_feature_backtest(
        candidate_games,
        feature_columns=baseline_columns,
        start_season=selection_start_season,
        end_season=selection_end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )
    weekly_selection_report, _ = _run_weekly_refit_backtest(
        candidate_games,
        feature_columns=baseline_columns,
        start_season=selection_start_season,
        end_season=selection_end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )
    frozen_selection = _model_summary(frozen_selection_report)
    weekly_selection = _model_summary(weekly_selection_report)
    selection_loss = _normalized_selection_loss(weekly_selection, frozen_selection)

    frozen_validation_report, _ = _run_feature_backtest(
        candidate_games,
        feature_columns=baseline_columns,
        start_season=validation_season,
        end_season=validation_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state + 100,
    )
    weekly_validation_report, weekly_predictions = _run_weekly_refit_backtest(
        candidate_games,
        feature_columns=baseline_columns,
        start_season=validation_season,
        end_season=validation_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state + 100,
    )
    frozen_validation = _model_summary(frozen_validation_report)
    weekly_validation = _model_summary(weekly_validation_report)
    validation_loss = _normalized_selection_loss(weekly_validation, frozen_validation)

    selection_improves = bool(np.isfinite(selection_loss) and selection_loss < 1.0)
    validation_improves = bool(np.isfinite(validation_loss) and validation_loss < 1.0)
    validation_point_nonworse = _primary_nonworse(weekly_validation, frozen_validation)
    passes_gate = bool(selection_improves and validation_improves and validation_point_nonworse)

    report: dict[str, object] = {
        "selection_window": {
            "start_season": int(selection_start_season),
            "end_season": int(selection_end_season),
        },
        "validation_season": int(validation_season),
        "reserved_confirmation_starts": DEVELOPMENT_MAX_SEASON + 1,
        "baseline_policy": "78-feature deployable pregame-only defaults",
        "candidate_policy": (
            "same 78 features, refit at each NFL week boundary using completed earlier weeks"
        ),
        "feature_count": len(baseline_columns),
        "selection_metrics": list(SELECTION_METRICS),
        "development_gate": {
            "requirements": [
                "balanced_selection_loss_improves",
                "balanced_validation_loss_improves",
                "validation_margin_mae_not_worse",
                "validation_total_mae_not_worse",
            ],
            "passes": passes_gate,
        },
        "selection": {
            "season_frozen_summary": frozen_selection,
            "weekly_refit_summary": weekly_selection,
            "weekly_vs_frozen_improvement_pct": _comparison_improvements(
                weekly_selection,
                frozen_selection,
            ),
            "normalized_loss_ratio": selection_loss,
            "balanced_improvement_pct": (1.0 - selection_loss) * 100.0,
            "weekly_by_season": weekly_selection_report["by_season"],
            "weekly_refits": weekly_selection_report["summary"].get("refits"),
        },
        "validation": {
            "season_frozen_summary": frozen_validation,
            "weekly_refit_summary": weekly_validation,
            "weekly_vs_frozen_improvement_pct": _comparison_improvements(
                weekly_validation,
                frozen_validation,
            ),
            "normalized_loss_ratio": validation_loss,
            "balanced_improvement_pct": (1.0 - validation_loss) * 100.0,
            "weekly_by_season": weekly_validation_report["by_season"],
            "weekly_refits": weekly_validation_report["summary"].get("refits"),
        },
        "selection_skipped": weekly_selection_report["skipped"],
        "validation_skipped": weekly_validation_report["skipped"],
    }
    return report, weekly_predictions
