from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error

from .model import NFLPredictor


def _expected_calibration_error(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Weighted absolute calibration error for binary probabilities."""
    if len(outcomes) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), bins - 1)
    total = len(outcomes)
    error = 0.0
    for i in range(bins):
        mask = bucket == i
        if not mask.any():
            continue
        error += (mask.sum() / total) * abs(float(outcomes[mask].mean() - probabilities[mask].mean()))
    return float(error)


def score_holdout(
    model: NFLPredictor,
    games: pd.DataFrame,
    max_distribution_games: int | None = None,
) -> pd.DataFrame:
    """Generate row-level out-of-sample predictions for completed games."""
    completed = games.loc[games["home_score"].notna() & games["away_score"].notna()].copy()
    if completed.empty:
        raise ValueError("holdout contains no completed games")

    rows: list[dict[str, float | int | str]] = []
    for i, (_, row) in enumerate(completed.iterrows()):
        point = model.predict_point(row)
        dist = model.predict_distribution(row)
        home_score = float(row["home_score"])
        away_score = float(row["away_score"])
        exact_probability = float("nan")
        if max_distribution_games is None or i < max_distribution_games:
            exact_probability = dist.exact_score(int(home_score), int(away_score))
        home_win = float("nan") if home_score == away_score else float(home_score > away_score)
        home_win_probability = dist.moneyline("home").fair_probability

        result: dict[str, float | int | str] = {
            "home_score": home_score,
            "away_score": away_score,
            "actual_margin": home_score - away_score,
            "actual_total": home_score + away_score,
            "predicted_margin": point.predicted_margin,
            "predicted_total": point.predicted_total,
            "predicted_home_score": point.predicted_home_score,
            "predicted_away_score": point.predicted_away_score,
            "home_win": home_win,
            "home_win_probability": home_win_probability,
            "exact_score_probability": exact_probability,
        }
        for column in ("game_id", "season", "week", "game_date", "home_team", "away_team"):
            if column in row.index and pd.notna(row[column]):
                result[column] = row[column]
        rows.append(result)
    return pd.DataFrame(rows)


def metrics_from_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute point, binary-probability, and exact-score metrics."""
    if predictions.empty:
        raise ValueError("predictions are empty")

    actual_margin = predictions["actual_margin"].astype(float).to_numpy()
    actual_total = predictions["actual_total"].astype(float).to_numpy()
    pred_margin = predictions["predicted_margin"].astype(float).to_numpy()
    pred_total = predictions["predicted_total"].astype(float).to_numpy()

    win_rows = predictions.loc[
        predictions["home_win"].notna() & predictions["home_win_probability"].notna()
    ]
    if len(win_rows):
        win_y = win_rows["home_win"].astype(float).to_numpy()
        win_p = np.clip(win_rows["home_win_probability"].astype(float).to_numpy(), 1e-9, 1 - 1e-9)
        brier = float(brier_score_loss(win_y, win_p))
        log_loss = float(np.mean(-(win_y * np.log(win_p) + (1.0 - win_y) * np.log(1.0 - win_p))))
        ece = _expected_calibration_error(win_y, win_p)
    else:
        brier = log_loss = ece = float("nan")

    exact = predictions["exact_score_probability"].dropna().astype(float).to_numpy()
    exact_nll = float(np.mean(-np.log(np.maximum(exact, 1e-12)))) if len(exact) else float("nan")

    return {
        "games": float(len(predictions)),
        "margin_mae": float(mean_absolute_error(actual_margin, pred_margin)),
        "margin_rmse": float(mean_squared_error(actual_margin, pred_margin) ** 0.5),
        "total_mae": float(mean_absolute_error(actual_total, pred_total)),
        "total_rmse": float(mean_squared_error(actual_total, pred_total) ** 0.5),
        "home_win_brier": brier,
        "home_win_log_loss": log_loss,
        "home_win_ece": ece,
        "exact_score_nll": exact_nll,
    }


def evaluate_holdout(
    model: NFLPredictor,
    games: pd.DataFrame,
    max_distribution_games: int = 400,
) -> dict[str, float]:
    """Evaluate a fitted model on chronologically later completed games."""
    predictions = score_holdout(
        model,
        games,
        max_distribution_games=max_distribution_games,
    )
    return metrics_from_predictions(predictions)
