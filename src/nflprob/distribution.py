from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .pricing import MarketPrice, market_price


@dataclass(frozen=True)
class JointScoreDistribution:
    """Discrete joint distribution P(home_score, away_score)."""

    probabilities: np.ndarray

    def __post_init__(self) -> None:
        probs = np.asarray(self.probabilities, dtype=float)
        if probs.ndim != 2 or probs.shape[0] != probs.shape[1]:
            raise ValueError("probabilities must be a square 2-D array")
        if np.any(probs < -1e-12):
            raise ValueError("probabilities cannot be negative")
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("probabilities must have positive finite mass")
        object.__setattr__(self, "probabilities", np.maximum(probs, 0.0) / total)

    @property
    def score_max(self) -> int:
        return self.probabilities.shape[0] - 1

    @property
    def expected_home(self) -> float:
        scores = np.arange(self.score_max + 1, dtype=float)
        return float((self.probabilities * scores[:, None]).sum())

    @property
    def expected_away(self) -> float:
        scores = np.arange(self.score_max + 1, dtype=float)
        return float((self.probabilities * scores[None, :]).sum())

    @property
    def expected_margin(self) -> float:
        return self.expected_home - self.expected_away

    @property
    def expected_total(self) -> float:
        return self.expected_home + self.expected_away

    def exact_score(self, home_score: int, away_score: int) -> float:
        if not (0 <= home_score <= self.score_max and 0 <= away_score <= self.score_max):
            return 0.0
        return float(self.probabilities[home_score, away_score])

    def moneyline(self, side: str = "home", push_on_tie: bool = True) -> MarketPrice:
        h, a = np.indices(self.probabilities.shape)
        if side.lower() == "home":
            win_mask, lose_mask = h > a, h < a
        elif side.lower() == "away":
            win_mask, lose_mask = a > h, a < h
        else:
            raise ValueError("side must be 'home' or 'away'")
        tie = float(self.probabilities[h == a].sum())
        win = float(self.probabilities[win_mask].sum())
        lose = float(self.probabilities[lose_mask].sum())
        if push_on_tie:
            return market_price(win, tie, lose)
        return market_price(win, 0.0, lose + tie)

    def spread(self, home_line: float, side: str = "home") -> MarketPrice:
        """Price a point spread using the home-team line convention."""
        h, a = np.indices(self.probabilities.shape)
        graded = (h - a) + float(home_line)
        push_mask = np.isclose(graded, 0.0, atol=1e-12)
        home_win = graded > 0.0
        home_lose = graded < 0.0
        if side.lower() == "home":
            win_mask, lose_mask = home_win, home_lose
        elif side.lower() == "away":
            win_mask, lose_mask = home_lose, home_win
        else:
            raise ValueError("side must be 'home' or 'away'")
        return market_price(
            float(self.probabilities[win_mask].sum()),
            float(self.probabilities[push_mask].sum()),
            float(self.probabilities[lose_mask].sum()),
        )

    def total(self, line: float, side: str = "over") -> MarketPrice:
        h, a = np.indices(self.probabilities.shape)
        graded = (h + a) - float(line)
        push_mask = np.isclose(graded, 0.0, atol=1e-12)
        over_mask = graded > 0.0
        under_mask = graded < 0.0
        if side.lower() == "over":
            win_mask, lose_mask = over_mask, under_mask
        elif side.lower() == "under":
            win_mask, lose_mask = under_mask, over_mask
        else:
            raise ValueError("side must be 'over' or 'under'")
        return market_price(
            float(self.probabilities[win_mask].sum()),
            float(self.probabilities[push_mask].sum()),
            float(self.probabilities[lose_mask].sum()),
        )

    def margin_distribution(self) -> pd.Series:
        h, a = np.indices(self.probabilities.shape)
        margins = (h - a).ravel()
        probs = self.probabilities.ravel()
        out = pd.Series(probs).groupby(margins).sum()
        out.index.name = "home_margin"
        out.name = "probability"
        return out.sort_index()

    def total_distribution(self) -> pd.Series:
        h, a = np.indices(self.probabilities.shape)
        totals = (h + a).ravel()
        probs = self.probabilities.ravel()
        out = pd.Series(probs).groupby(totals).sum()
        out.index.name = "game_total"
        out.name = "probability"
        return out.sort_index()

    def to_frame(self, min_probability: float = 0.0) -> pd.DataFrame:
        h, a = np.indices(self.probabilities.shape)
        frame = pd.DataFrame(
            {
                "home_score": h.ravel(),
                "away_score": a.ravel(),
                "probability": self.probabilities.ravel(),
            }
        )
        if min_probability > 0:
            frame = frame.loc[frame["probability"] >= min_probability]
        return frame.sort_values("probability", ascending=False, ignore_index=True)


class ScoreDistributionCalibrator:
    """Analog-conditioned, maximum-entropy score-distribution calibrator."""

    def __init__(
        self,
        score_max: int = 80,
        neighbors: int = 900,
        margin_bandwidth: float = 7.0,
        total_bandwidth: float = 10.0,
        prior_fraction: float = 0.04,
        recency_half_life_seasons: float = 5.0,
        temperature: float = 1.0,
    ) -> None:
        self.score_max = int(score_max)
        self.neighbors = int(neighbors)
        self.margin_bandwidth = float(margin_bandwidth)
        self.total_bandwidth = float(total_bandwidth)
        self.prior_fraction = float(prior_fraction)
        self.recency_half_life_seasons = float(recency_half_life_seasons)
        self.temperature = float(temperature)
        self._fitted = False

    def fit(
        self,
        home_scores: np.ndarray,
        away_scores: np.ndarray,
        predicted_margins: np.ndarray,
        predicted_totals: np.ndarray,
        seasons: np.ndarray | None = None,
    ) -> "ScoreDistributionCalibrator":
        hs = np.asarray(home_scores, dtype=int)
        aw = np.asarray(away_scores, dtype=int)
        pm = np.asarray(predicted_margins, dtype=float)
        pt = np.asarray(predicted_totals, dtype=float)
        valid = (
            np.isfinite(hs)
            & np.isfinite(aw)
            & np.isfinite(pm)
            & np.isfinite(pt)
            & (hs >= 0)
            & (aw >= 0)
        )
        if valid.sum() < 50:
            raise ValueError("at least 50 valid out-of-fold games are required")
        self.home_scores_ = np.clip(hs[valid], 0, self.score_max)
        self.away_scores_ = np.clip(aw[valid], 0, self.score_max)
        self.predicted_margins_ = pm[valid]
        self.predicted_totals_ = pt[valid]
        if seasons is None:
            self.seasons_ = np.zeros(valid.sum(), dtype=float)
        else:
            self.seasons_ = np.asarray(seasons, dtype=float)[valid]

        all_scores = np.concatenate([self.home_scores_, self.away_scores_])
        counts = np.bincount(all_scores, minlength=self.score_max + 1).astype(float)
        counts += 0.15
        self.global_score_prior_ = counts / counts.sum()
        self._fitted = True
        return self

    def predict(
        self,
        predicted_margin: float,
        predicted_total: float,
        season: int | None = None,
    ) -> JointScoreDistribution:
        if not self._fitted:
            raise RuntimeError("calibrator must be fitted before prediction")
        home_mean = (float(predicted_total) + float(predicted_margin)) / 2.0
        away_mean = (float(predicted_total) - float(predicted_margin)) / 2.0
        home_mean = float(np.clip(home_mean, 0.25, self.score_max - 0.25))
        away_mean = float(np.clip(away_mean, 0.25, self.score_max - 0.25))

        dm = (self.predicted_margins_ - float(predicted_margin)) / self.margin_bandwidth
        dt = (self.predicted_totals_ - float(predicted_total)) / self.total_bandwidth
        distance = dm * dm + dt * dt
        k = min(self.neighbors, len(distance))
        if k < len(distance):
            idx = np.argpartition(distance, k - 1)[:k]
        else:
            idx = np.arange(len(distance))
        weights = np.exp(-0.5 * distance[idx])

        if season is not None and np.any(self.seasons_ != 0):
            age = np.maximum(float(season) - self.seasons_[idx], 0.0)
            weights *= np.exp(-np.log(2.0) * age / self.recency_half_life_seasons)

        base = np.zeros((self.score_max + 1, self.score_max + 1), dtype=float)
        np.add.at(base, (self.home_scores_[idx], self.away_scores_[idx]), weights)
        mass = max(float(base.sum()), 1.0)
        base += self.prior_fraction * mass * np.outer(
            self.global_score_prior_, self.global_score_prior_
        )
        base += np.finfo(float).tiny
        base /= base.sum()

        if self.temperature != 1.0:
            log_base = np.log(base) / self.temperature
            base = np.exp(log_base - logsumexp(log_base))

        return JointScoreDistribution(self._tilt_to_means(base, home_mean, away_mean))

    def _tilt_to_means(self, base: np.ndarray, home_mean: float, away_mean: float) -> np.ndarray:
        scores = np.arange(self.score_max + 1, dtype=float)
        home_grid = np.broadcast_to(scores[:, None], base.shape)
        away_grid = np.broadcast_to(scores[None, :], base.shape)
        log_base = np.log(base)

        def objective(lam: np.ndarray) -> tuple[float, np.ndarray]:
            logits = log_base + lam[0] * home_grid + lam[1] * away_grid
            log_z = logsumexp(logits)
            probs = np.exp(logits - log_z)
            eh = float((probs * home_grid).sum())
            ea = float((probs * away_grid).sum())
            value = log_z - lam[0] * home_mean - lam[1] * away_mean
            return value, np.array([eh - home_mean, ea - away_mean])

        result = minimize(
            lambda x: objective(x)[0],
            x0=np.zeros(2),
            jac=lambda x: objective(x)[1],
            method="L-BFGS-B",
            bounds=[(-1.5, 1.5), (-1.5, 1.5)],
            options={"maxiter": 100, "ftol": 1e-12},
        )
        lam = result.x if result.success else np.zeros(2)
        logits = log_base + lam[0] * home_grid + lam[1] * away_grid
        return np.exp(logits - logsumexp(logits))
