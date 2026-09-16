from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error

from .model import NFLPredictor


def evaluate_holdout(
    model: NFLPredictor,
    games: pd.DataFrame,
    max_distribution_games: int = 400,
) -> dict[str, float]:
    """Evaluate a fitted model on chronologically later completed games."""
    completed = games.loc[games["home_score"].notna() & games["away_score"].notna()].copy()
    if completed.empty:
        raise ValueError("holdout contains no completed games")

    actual_margin: list[float] = []
    actual_total: list[float] = []
    pred_margin: list[float] = []
    pred_total: list[float] = []
    win_y: list[float] = []
    win_p: list[float] = []
    nll: list[float] = []

    for i, (_, row) in enumerate(completed.iterrows()):
        point = model.predict_point(row)
        actual_margin.append(float(row["home_score"] - row["away_score"]))
        actual_total.append(float(row["home_score"] + row["away_score"]))
        pred_margin.append(point.predicted_margin)
        pred_total.append(point.predicted_total)
        if row["home_score"] != row["away_score"]:
            dist = model.predict_distribution(row)
            win_p.append(dist.moneyline("home").fair_probability)
            win_y.append(float(row["home_score"] > row["away_score"]))
        if i < max_distribution_games:
            dist = model.predict_distribution(row)
            p = dist.exact_score(int(row["home_score"]), int(row["away_score"]))
            nll.append(-math.log(max(p, 1e-12)))

    return {
        "games": float(len(completed)),
        "margin_mae": float(mean_absolute_error(actual_margin, pred_margin)),
        "margin_rmse": float(mean_squared_error(actual_margin, pred_margin) ** 0.5),
        "total_mae": float(mean_absolute_error(actual_total, pred_total)),
        "total_rmse": float(mean_squared_error(actual_total, pred_total) ** 0.5),
        "home_win_brier": float(brier_score_loss(win_y, win_p)) if win_y else float("nan"),
        "exact_score_nll": float(np.mean(nll)) if nll else float("nan"),
    }
