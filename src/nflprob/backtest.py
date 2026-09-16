from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error

from .evaluation import metrics_from_predictions, score_holdout
from .model import NFLPredictor


def _baseline_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Metrics for the expanding-history constant baseline attached to each row."""
    actual_margin = predictions["actual_margin"].astype(float).to_numpy()
    actual_total = predictions["actual_total"].astype(float).to_numpy()
    margin = predictions["baseline_margin"].astype(float).to_numpy()
    total = predictions["baseline_total"].astype(float).to_numpy()
    win_rows = predictions.loc[
        predictions["home_win"].notna() & predictions["baseline_home_win_probability"].notna()
    ]
    brier = float("nan")
    if len(win_rows):
        brier = float(
            brier_score_loss(
                win_rows["home_win"].astype(float),
                win_rows["baseline_home_win_probability"].astype(float),
            )
        )
    return {
        "margin_mae": float(mean_absolute_error(actual_margin, margin)),
        "margin_rmse": float(mean_squared_error(actual_margin, margin) ** 0.5),
        "total_mae": float(mean_absolute_error(actual_total, total)),
        "total_rmse": float(mean_squared_error(actual_total, total) ** 0.5),
        "home_win_brier": brier,
    }


def _relative_improvement(model_value: float, baseline_value: float) -> float:
    if not np.isfinite(model_value) or not np.isfinite(baseline_value) or baseline_value == 0:
        return float("nan")
    return float((baseline_value - model_value) / baseline_value * 100.0)


def _attach_comparison_metrics(
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, float]:
    output = dict(model_metrics)
    for name, value in baseline_metrics.items():
        output[f"baseline_{name}"] = value
    for name in ("margin_mae", "margin_rmse", "total_mae", "total_rmse", "home_win_brier"):
        output[f"{name}_improvement_pct"] = _relative_improvement(
            model_metrics[name], baseline_metrics[name]
        )
    return output


def walk_forward_backtest(
    games: pd.DataFrame,
    *,
    start_season: int,
    end_season: int | None = None,
    min_train_games: int = 500,
    score_max: int = 80,
    random_state: int = 7,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Refit on prior seasons and evaluate each requested season out of sample.

    The split is deliberately coarse and auditable: test season S is predicted only by a
    model fit on completed games with season < S. This prevents within-season lookahead.
    """
    required = {"season", "home_score", "away_score"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"backtest data missing columns: {sorted(missing)}")

    completed = games.loc[games["home_score"].notna() & games["away_score"].notna()].copy()
    completed["season"] = pd.to_numeric(completed["season"], errors="raise").astype(int)
    sort_columns = [c for c in ("season", "week", "game_date", "game_id") if c in completed.columns]
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
        raise ValueError("no completed games exist in the requested backtest seasons")

    season_reports: list[dict[str, float]] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, int]] = []

    for offset, season in enumerate(requested):
        train = completed.loc[completed["season"] < season].copy()
        test = completed.loc[completed["season"] == season].copy()
        if len(train) < min_train_games:
            skipped.append({
                "season": int(season),
                "train_games": int(len(train)),
                "required_train_games": int(min_train_games),
            })
            continue
        if test.empty:
            continue

        model = NFLPredictor(
            random_state=random_state + offset,
            score_max=score_max,
        ).fit(train)
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
        predictions["train_games"] = int(len(train))
        prediction_frames.append(predictions)

        model_metrics = metrics_from_predictions(predictions)
        baseline_metrics = _baseline_metrics(predictions)
        report = _attach_comparison_metrics(model_metrics, baseline_metrics)
        report["season"] = float(season)
        report["train_games"] = float(len(train))
        season_reports.append(report)

    if not prediction_frames:
        raise ValueError(
            "no seasons could be backtested; lower --min-train-games or choose later seasons"
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    summary = _attach_comparison_metrics(
        metrics_from_predictions(all_predictions),
        _baseline_metrics(all_predictions),
    )
    summary["first_test_season"] = float(all_predictions["backtest_season"].min())
    summary["last_test_season"] = float(all_predictions["backtest_season"].max())
    summary["seasons_tested"] = float(all_predictions["backtest_season"].nunique())

    return {
        "summary": summary,
        "by_season": season_reports,
        "skipped": skipped,
    }, all_predictions
