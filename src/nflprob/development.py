from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import metrics_from_predictions, score_holdout
from .features import numeric_feature_columns
from .model import NFLPredictor, is_experimental_feature


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


def pace_scoring_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return engineered feature columns added by the pace/scoring experiment."""
    selected = numeric_feature_columns(frame, extra_exclude={"season", "week"})
    return [column for column in selected if is_experimental_feature(column)]


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
    baseline_columns = [column for column in all_columns if not is_experimental_feature(column)]
    candidate_columns = all_columns

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
