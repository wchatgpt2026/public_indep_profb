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


def _american_implied_probability(odds: np.ndarray) -> np.ndarray:
    """Convert American odds to raw implied probability before vig removal."""
    values = np.asarray(odds, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    positive = np.isfinite(values) & (values > 0)
    negative = np.isfinite(values) & (values < 0)
    out[positive] = 100.0 / (values[positive] + 100.0)
    out[negative] = (-values[negative]) / ((-values[negative]) + 100.0)
    return out


def _binary_log_loss(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def _market_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Compare the independent model with evaluation-only nflverse market data."""
    output: dict[str, float] = {}

    if "spread_line" in predictions.columns:
        spread = pd.to_numeric(predictions["spread_line"], errors="coerce").to_numpy(dtype=float)
        actual = predictions["actual_margin"].astype(float).to_numpy()
        model = predictions["predicted_margin"].astype(float).to_numpy()
        mask = np.isfinite(spread) & np.isfinite(actual) & np.isfinite(model)
        if mask.any():
            market_mae = float(mean_absolute_error(actual[mask], spread[mask]))
            market_rmse = float(mean_squared_error(actual[mask], spread[mask]) ** 0.5)
            model_mae = float(mean_absolute_error(actual[mask], model[mask]))
            model_rmse = float(mean_squared_error(actual[mask], model[mask]) ** 0.5)
            output.update(
                {
                    "market_margin_games": float(mask.sum()),
                    "market_margin_mae": market_mae,
                    "market_margin_rmse": market_rmse,
                    "model_margin_mae_same_games": model_mae,
                    "model_margin_rmse_same_games": model_rmse,
                    "margin_mae_vs_market_pct": _relative_improvement(model_mae, market_mae),
                    "margin_rmse_vs_market_pct": _relative_improvement(model_rmse, market_rmse),
                }
            )

    if "total_line" in predictions.columns:
        line = pd.to_numeric(predictions["total_line"], errors="coerce").to_numpy(dtype=float)
        actual = predictions["actual_total"].astype(float).to_numpy()
        model = predictions["predicted_total"].astype(float).to_numpy()
        mask = np.isfinite(line) & np.isfinite(actual) & np.isfinite(model)
        if mask.any():
            market_mae = float(mean_absolute_error(actual[mask], line[mask]))
            market_rmse = float(mean_squared_error(actual[mask], line[mask]) ** 0.5)
            model_mae = float(mean_absolute_error(actual[mask], model[mask]))
            model_rmse = float(mean_squared_error(actual[mask], model[mask]) ** 0.5)
            output.update(
                {
                    "market_total_games": float(mask.sum()),
                    "market_total_mae": market_mae,
                    "market_total_rmse": market_rmse,
                    "model_total_mae_same_games": model_mae,
                    "model_total_rmse_same_games": model_rmse,
                    "total_mae_vs_market_pct": _relative_improvement(model_mae, market_mae),
                    "total_rmse_vs_market_pct": _relative_improvement(model_rmse, market_rmse),
                }
            )

    moneyline_columns = {"home_moneyline", "away_moneyline"}
    if moneyline_columns.issubset(predictions.columns):
        home_odds = pd.to_numeric(predictions["home_moneyline"], errors="coerce").to_numpy(
            dtype=float
        )
        away_odds = pd.to_numeric(predictions["away_moneyline"], errors="coerce").to_numpy(
            dtype=float
        )
        home_raw = _american_implied_probability(home_odds)
        away_raw = _american_implied_probability(away_odds)
        denominator = home_raw + away_raw
        market_probability = home_raw / denominator
        outcomes = predictions["home_win"].astype(float).to_numpy()
        model_probability = predictions["home_win_probability"].astype(float).to_numpy()
        mask = (
            np.isfinite(outcomes)
            & np.isfinite(model_probability)
            & np.isfinite(market_probability)
            & np.isfinite(denominator)
            & (denominator > 0.0)
        )
        if mask.any():
            y = outcomes[mask]
            market_p = market_probability[mask]
            model_p = model_probability[mask]
            market_brier = float(brier_score_loss(y, market_p))
            model_brier = float(brier_score_loss(y, model_p))
            market_log_loss = _binary_log_loss(y, market_p)
            model_log_loss = _binary_log_loss(y, model_p)
            output.update(
                {
                    "market_moneyline_games": float(mask.sum()),
                    "market_home_win_brier": market_brier,
                    "market_home_win_log_loss": market_log_loss,
                    "model_home_win_brier_same_games": model_brier,
                    "model_home_win_log_loss_same_games": model_log_loss,
                    "home_win_brier_vs_market_pct": _relative_improvement(
                        model_brier, market_brier
                    ),
                    "home_win_log_loss_vs_market_pct": _relative_improvement(
                        model_log_loss, market_log_loss
                    ),
                    "market_moneyline_mean_overround": float(
                        np.mean(denominator[mask] - 1.0)
                    ),
                }
            )

    return output


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
    Sportsbook fields are retained only for post-prediction benchmarking and are blocked
    from NFLPredictor's feature matrix.
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
        predictions["train_games"] = len(train)
        prediction_frames.append(predictions)

        model_metrics = metrics_from_predictions(predictions)
        baseline_metrics = _baseline_metrics(predictions)
        report = _attach_comparison_metrics(model_metrics, baseline_metrics)
        report.update(_market_metrics(predictions))
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
    summary.update(_market_metrics(all_predictions))
    summary["first_test_season"] = float(all_predictions["backtest_season"].min())
    summary["last_test_season"] = float(all_predictions["backtest_season"].max())
    summary["seasons_tested"] = float(all_predictions["backtest_season"].nunique())

    return {
        "summary": summary,
        "by_season": season_reports,
        "skipped": skipped,
    }, all_predictions
