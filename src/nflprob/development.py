from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .evaluation import metrics_from_predictions, score_holdout
from .features import numeric_feature_columns
from .model import NFLPredictor, default_feature_columns, is_experimental_feature


DEVELOPMENT_MAX_SEASON = 2021
MARKET_TOKENS = ("moneyline", "spread_line", "total_line", "odds", "vegas", "market_")
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
SELECTION_METRICS = (
    "margin_mae",
    "total_mae",
    "home_win_brier",
    "exact_score_nll",
)
EXPERIMENT_GROUP_METRICS = {
    "pace_volume": (
        "off_plays",
        "off_drives",
        "off_plays_per_drive",
        "off_seconds_per_play",
        "def_plays_faced",
        "def_drives_faced",
        "def_plays_per_drive_allowed",
        "def_seconds_per_play_allowed",
    ),
    "play_calling": (
        "off_no_huddle_rate",
        "off_early_down_pass_rate",
        "def_no_huddle_rate_allowed",
        "def_early_down_pass_rate_allowed",
    ),
    "scoring_efficiency": (
        "off_red_zone_epa",
        "off_red_zone_success_rate",
        "off_scoring_drive_rate",
        "def_red_zone_epa_allowed",
        "def_red_zone_success_allowed",
        "def_scoring_drive_rate_allowed",
    ),
}


def pace_scoring_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return engineered feature columns added by the pace/scoring experiment."""
    selected = numeric_feature_columns(frame, extra_exclude={"season", "week"})
    return [column for column in selected if is_experimental_feature(column)]


def experimental_feature_group(column: str) -> str | None:
    """Map one engineered experimental column into an interpretable feature block."""
    for group, metrics in EXPERIMENT_GROUP_METRICS.items():
        if any(column.endswith(metric) for metric in metrics):
            return group
    return None


def _strip_market_columns(frame: pd.DataFrame) -> pd.DataFrame:
    blocked = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in MARKET_TOKENS)
    ]
    return frame.drop(columns=blocked, errors="ignore").copy()


def _run_feature_backtest(
    games: pd.DataFrame,
    *,
    feature_columns: list[str],
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

        model = NFLPredictor(
            random_state=random_state + offset,
            score_max=score_max,
        ).fit(train, feature_columns=feature_columns)
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


def _comparison_improvements(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        metric: _improvement_pct(candidate[metric], baseline[metric])
        for metric in COMPARISON_METRICS
        if metric in baseline and metric in candidate
    }


def _normalized_selection_loss(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> float:
    ratios: list[float] = []
    for metric in SELECTION_METRICS:
        c = candidate.get(metric, float("nan"))
        b = baseline.get(metric, float("nan"))
        if np.isfinite(c) and np.isfinite(b) and b > 0.0:
            ratios.append(c / b)
    if len(ratios) != len(SELECTION_METRICS):
        return float("inf")
    return float(np.mean(ratios))


def _experimental_columns_by_group(all_columns: list[str]) -> dict[str, list[str]]:
    groups = {name: [] for name in EXPERIMENT_GROUP_METRICS}
    for column in all_columns:
        if not is_experimental_feature(column):
            continue
        group = experimental_feature_group(column)
        if group is not None:
            groups[group].append(column)
    return groups


def development_feature_compare(
    games: pd.DataFrame,
    *,
    start_season: int = 2019,
    end_season: int = DEVELOPMENT_MAX_SEASON,
    min_train_games: int = 500,
    score_max: int = 80,
    random_state: int = 7,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare operational defaults vs all pace/scoring features on development seasons."""
    if end_season > DEVELOPMENT_MAX_SEASON:
        raise ValueError(
            "dev-compare is locked to seasons through 2021; use the regular backtest only "
            "after a development candidate has been selected"
        )
    if start_season > end_season:
        raise ValueError("start_season must be <= end_season")

    candidate_games = _strip_market_columns(games)
    all_columns = numeric_feature_columns(
        candidate_games,
        extra_exclude={"season", "week"},
    )
    added_features = [column for column in all_columns if is_experimental_feature(column)]
    if not added_features:
        raise ValueError(
            "no pace/scoring feature columns were found; rebuild the dataset with the current "
            "`nflprob build-data` command before running dev-compare"
        )
    baseline_columns = default_feature_columns(candidate_games)
    candidate_columns = baseline_columns + added_features

    baseline_report, _ = _run_feature_backtest(
        candidate_games,
        feature_columns=baseline_columns,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )
    candidate_report, candidate_predictions = _run_feature_backtest(
        candidate_games,
        feature_columns=candidate_columns,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )

    baseline_summary = _model_summary(baseline_report)
    candidate_summary = _model_summary(candidate_report)

    report: dict[str, object] = {
        "development_window": {
            "start_season": int(start_season),
            "end_season": int(end_season),
            "reserved_confirmation_starts": DEVELOPMENT_MAX_SEASON + 1,
        },
        "baseline_policy": "deployable pregame-only defaults",
        "feature_counts": {
            "baseline": len(baseline_columns),
            "candidate": len(candidate_columns),
            "added": len(added_features),
        },
        "added_features": added_features,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "candidate_vs_baseline_improvement_pct": _comparison_improvements(
            candidate_summary,
            baseline_summary,
        ),
        "baseline_by_season": baseline_report["by_season"],
        "candidate_by_season": candidate_report["by_season"],
        "baseline_skipped": baseline_report["skipped"],
        "candidate_skipped": candidate_report["skipped"],
    }
    return report, candidate_predictions


def development_group_ablation(
    games: pd.DataFrame,
    *,
    selection_start_season: int = 2019,
    selection_end_season: int = 2020,
    validation_season: int = 2021,
    min_train_games: int = 500,
    score_max: int = 80,
    random_state: int = 7,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Select experimental groups against the operational baseline, validate on 2021.

    All sportsbook fields are removed before fitting or scoring. The feature-group winner is
    chosen using 2019-2020 only, then evaluated once against the deployable pregame-only default
    feature set on 2021. Seasons 2022+ remain outside this research-selection loop.
    """
    if validation_season > DEVELOPMENT_MAX_SEASON:
        raise ValueError("dev-ablate validation is locked to seasons through 2021")
    if selection_end_season >= validation_season:
        raise ValueError("selection_end_season must be earlier than validation_season")
    if selection_start_season > selection_end_season:
        raise ValueError("selection_start_season must be <= selection_end_season")

    candidate_games = _strip_market_columns(games)
    all_columns = numeric_feature_columns(
        candidate_games,
        extra_exclude={"season", "week"},
    )
    baseline_columns = default_feature_columns(candidate_games)
    group_columns = _experimental_columns_by_group(all_columns)
    missing_groups = [name for name, columns in group_columns.items() if not columns]
    if missing_groups:
        raise ValueError(
            "experimental feature groups are missing from the dataset: " + ", ".join(missing_groups)
        )

    baseline_selection_report, _ = _run_feature_backtest(
        candidate_games,
        feature_columns=baseline_columns,
        start_season=selection_start_season,
        end_season=selection_end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )
    baseline_selection = _model_summary(baseline_selection_report)

    candidate_results: list[dict[str, object]] = []
    best_groups: tuple[str, ...] = ()
    best_loss = 1.0
    best_added_count = 0
    group_names = tuple(EXPERIMENT_GROUP_METRICS)

    for size in range(1, len(group_names) + 1):
        for selected_groups in combinations(group_names, size):
            added_columns = [
                column
                for group in selected_groups
                for column in group_columns[group]
            ]
            report, _ = _run_feature_backtest(
                candidate_games,
                feature_columns=baseline_columns + added_columns,
                start_season=selection_start_season,
                end_season=selection_end_season,
                min_train_games=min_train_games,
                score_max=score_max,
                random_state=random_state,
            )
            summary = _model_summary(report)
            loss = _normalized_selection_loss(summary, baseline_selection)
            improvement = (1.0 - loss) * 100.0 if np.isfinite(loss) else float("nan")
            candidate_results.append(
                {
                    "groups": list(selected_groups),
                    "added_features": len(added_columns),
                    "selection_loss_ratio": loss,
                    "balanced_improvement_pct": improvement,
                    "summary": summary,
                    "vs_baseline_improvement_pct": _comparison_improvements(
                        summary,
                        baseline_selection,
                    ),
                }
            )
            if loss < best_loss - 1e-12 or (
                abs(loss - best_loss) <= 1e-12 and len(added_columns) < best_added_count
            ):
                best_loss = loss
                best_groups = selected_groups
                best_added_count = len(added_columns)

    winner_columns = [
        column
        for group in best_groups
        for column in group_columns[group]
    ]
    baseline_validation_report, _ = _run_feature_backtest(
        candidate_games,
        feature_columns=baseline_columns,
        start_season=validation_season,
        end_season=validation_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state + 100,
    )
    winner_validation_report, winner_predictions = _run_feature_backtest(
        candidate_games,
        feature_columns=baseline_columns + winner_columns,
        start_season=validation_season,
        end_season=validation_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state + 100,
    )
    baseline_validation = _model_summary(baseline_validation_report)
    winner_validation = _model_summary(winner_validation_report)

    candidate_results.sort(
        key=lambda row: (
            float(row["selection_loss_ratio"]),
            int(row["added_features"]),
        )
    )
    report: dict[str, object] = {
        "selection_window": {
            "start_season": int(selection_start_season),
            "end_season": int(selection_end_season),
        },
        "validation_season": int(validation_season),
        "reserved_confirmation_starts": DEVELOPMENT_MAX_SEASON + 1,
        "baseline_policy": "deployable pregame-only defaults",
        "selection_metrics": list(SELECTION_METRICS),
        "feature_groups": {
            name: {
                "features": columns,
                "count": len(columns),
            }
            for name, columns in group_columns.items()
        },
        "baseline_selection_summary": baseline_selection,
        "candidates": candidate_results,
        "winner": {
            "groups": list(best_groups),
            "added_features": len(winner_columns),
            "selection_loss_ratio": best_loss,
            "balanced_improvement_pct": (1.0 - best_loss) * 100.0,
        },
        "validation": {
            "baseline_summary": baseline_validation,
            "winner_summary": winner_validation,
            "winner_vs_baseline_improvement_pct": _comparison_improvements(
                winner_validation,
                baseline_validation,
            ),
        },
    }
    return report, winner_predictions
