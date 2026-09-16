from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketPrice:
    """Fair price for one side of a two-way market.

    ``fair_probability`` conditions on the bet not pushing. This makes the
    decimal/American price directly usable for integer spreads and totals.
    """

    win_probability: float
    push_probability: float
    lose_probability: float
    fair_probability: float
    fair_decimal: float
    fair_american: float

    def as_dict(self) -> dict[str, float]:
        return {
            "win_probability": self.win_probability,
            "push_probability": self.push_probability,
            "lose_probability": self.lose_probability,
            "fair_probability": self.fair_probability,
            "fair_decimal": self.fair_decimal,
            "fair_american": self.fair_american,
        }


def probability_to_decimal(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        if probability == 1.0:
            return 1.0
        return float("inf")
    return 1.0 / probability


def probability_to_american(probability: float) -> float:
    """Convert a fair probability to American odds."""
    if probability <= 0.0:
        return float("inf")
    if probability >= 1.0:
        return float("-inf")
    if np.isclose(probability, 0.5):
        return 100.0
    if probability > 0.5:
        return -100.0 * probability / (1.0 - probability)
    return 100.0 * (1.0 - probability) / probability


def market_price(win: float, push: float, lose: float) -> MarketPrice:
    win = float(np.clip(win, 0.0, 1.0))
    push = float(np.clip(push, 0.0, 1.0))
    lose = float(np.clip(lose, 0.0, 1.0))
    active = win + lose
    fair = win / active if active > 0 else 0.5
    return MarketPrice(
        win_probability=win,
        push_probability=push,
        lose_probability=lose,
        fair_probability=fair,
        fair_decimal=probability_to_decimal(fair),
        fair_american=probability_to_american(fair),
    )
