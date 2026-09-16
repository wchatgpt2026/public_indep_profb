from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .distribution import JointScoreDistribution, ScoreDistributionCalibrator
from .estimators import MarginTotalEnsemble
from .features import numeric_feature_columns


MARKET_TOKENS = ("moneyline", "spread_line", "total_line", "odds", "vegas", "market_")


@dataclass(frozen=True)
class PointPrediction:
    predicted_margin: float
    predicted_total: float

    @property
    def predicted_home_score(self) -> float:
        return (self.predicted_total + self.predicted_margin) / 2.0

    @property
    def predicted_away_score(self) -> float:
        return (self.predicted_total - self.predicted_margin) / 2.0


@dataclass(frozen=True)
class AffineTargetCalibrator:
    """Conservative affine correction learned from chronological OOF predictions.

    The correction is expressed around the OOF prediction mean so slope and level can be
    shrunk independently toward the identity mapping. That makes it useful for correcting
    systematic regression-to-the-mean without letting a noisy calibration window make large
    extrapolations.
    """

    center: float = 0.0
    scale: float = 1.0
    offset: float = 0.0

    @classmethod
    def fit(
        cls,
        predicted: np.ndarray,
        actual: np.ndarray,
        *,
        prior_strength: float = 250.0,
    ) -> AffineTargetCalibrator:
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        valid = np.isfinite(predicted) & np.isfinite(actual)
        predicted = predicted[valid]
        actual = actual[valid]
        if len(predicted) < 20:
            return cls()

        center = float(predicted.mean())
        pred_centered = predicted - center
        variance_mass = float(np.dot(pred_centered, pred_centered))
        if variance_mass <= np.finfo(float).eps:
            raw_scale = 1.0
        else:
            actual_centered = actual - float(actual.mean())
            raw_scale = float(np.dot(pred_centered, actual_centered) / variance_mass)

        reliability = float(len(predicted) / (len(predicted) + max(prior_strength, 0.0)))
        scale = 1.0 + reliability * (raw_scale - 1.0)
        scale = float(np.clip(scale, 0.75, 1.25))
        offset = reliability * float(np.mean(actual - predicted))
        return cls(center=center, scale=scale, offset=offset)

    def predict(self, values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return self.center + self.offset + self.scale * (values - self.center)


class NFLPredictor:
    """Independent NFL point model + calibrated full score distribution."""

    CALIBRATION_PRIOR_CANDIDATES = (0.0, 100.0, 250.0, 500.0, 1000.0, 1_000_000.0)

    def __init__(self, random_state: int = 7, score_max: int = 80, oof_folds: int = 4) -> None:
        self.random_state = int(random_state)
        self.score_max = int(score_max)
        self.oof_folds = int(oof_folds)
        self.feature_columns_: list[str] = []
        self.point_model_: MarginTotalEnsemble | None = None
        self.margin_calibrator_ = AffineTargetCalibrator()
        self.total_calibrator_ = AffineTargetCalibrator()
        self.distribution_: ScoreDistributionCalibrator | None = None
        self.training_metrics_: dict[str, float] = {}

    @staticmethod
    def _assert_independent(feature_columns: list[str]) -> None:
        offenders = [c for c in feature_columns if any(t in c.lower() for t in MARKET_TOKENS)]
        if offenders:
            raise ValueError(
                "Sportsbook/market features are blocked in independent mode: " + ", ".join(offenders)
            )

    @staticmethod
    def _sort_games(games: pd.DataFrame) -> pd.DataFrame:
        sort_cols = [c for c in ("game_date", "season", "week", "game_id") if c in games.columns]
        if sort_cols:
            return games.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        return games.reset_index(drop=True)

    @classmethod
    def _select_affine_calibrator(
        cls,
        predicted: np.ndarray,
        actual: np.ndarray,
    ) -> tuple[AffineTargetCalibrator, float, float]:
        """Tune shrinkage on a later OOF slice, then refit using all OOF rows."""
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        valid = np.isfinite(predicted) & np.isfinite(actual)
        predicted = predicted[valid]
        actual = actual[valid]
        if len(predicted) < 60:
            strength = 250.0
            return (
                AffineTargetCalibrator.fit(predicted, actual, prior_strength=strength),
                strength,
                float("nan"),
            )

        split = max(40, int(len(predicted) * 0.75))
        split = min(split, len(predicted) - 20)
        train_pred, validation_pred = predicted[:split], predicted[split:]
        train_actual, validation_actual = actual[:split], actual[split:]

        best_strength = cls.CALIBRATION_PRIOR_CANDIDATES[-1]
        best_mae = float("inf")
        for strength in cls.CALIBRATION_PRIOR_CANDIDATES:
            candidate = AffineTargetCalibrator.fit(
                train_pred,
                train_actual,
                prior_strength=strength,
            )
            score = float(mean_absolute_error(validation_actual, candidate.predict(validation_pred)))
            if score < best_mae - 1e-12 or (
                abs(score - best_mae) <= 1e-12 and strength > best_strength
            ):
                best_mae = score
                best_strength = strength

        return (
            AffineTargetCalibrator.fit(
                predicted,
                actual,
                prior_strength=best_strength,
            ),
            float(best_strength),
            best_mae,
        )

    def fit(self, games: pd.DataFrame, feature_columns: list[str] | None = None) -> NFLPredictor:
        required = {"home_score", "away_score"}
        missing = required - set(games.columns)
        if missing:
            raise ValueError(f"training data missing targets: {sorted(missing)}")
        train = games.loc[games["home_score"].notna() & games["away_score"].notna()].copy()
        train = self._sort_games(train)
        if len(train) < 120:
            raise ValueError("at least 120 completed games are required for fitting")
        if feature_columns is None:
            feature_columns = numeric_feature_columns(train, extra_exclude={"season", "week"})
        self._assert_independent(feature_columns)
        if not feature_columns:
            raise ValueError("no numeric feature columns were selected")
        self.feature_columns_ = list(feature_columns)
        X = train[self.feature_columns_].astype(float).to_numpy()
        home = train["home_score"].astype(float).to_numpy()
        away = train["away_score"].astype(float).to_numpy()
        actual_margin, actual_total = home - away, home + away

        raw_oof_margin, raw_oof_total, oof_mask = self._expanding_oof(X, home, away)
        if oof_mask.sum() < 50:
            raise ValueError("not enough chronological out-of-fold predictions for calibration")

        (
            self.margin_calibrator_,
            margin_prior_strength,
            margin_calibration_validation_mae,
        ) = self._select_affine_calibrator(
            raw_oof_margin[oof_mask],
            actual_margin[oof_mask],
        )
        (
            self.total_calibrator_,
            total_prior_strength,
            total_calibration_validation_mae,
        ) = self._select_affine_calibrator(
            raw_oof_total[oof_mask],
            actual_total[oof_mask],
        )
        oof_margin = self.margin_calibrator_.predict(raw_oof_margin[oof_mask])
        oof_total = self.total_calibrator_.predict(raw_oof_total[oof_mask])

        seasons = (
            train["season"].to_numpy(dtype=float)
            if "season" in train.columns
            else np.zeros(len(train), dtype=float)
        )
        self.distribution_ = ScoreDistributionCalibrator(score_max=self.score_max)
        self.distribution_.fit(
            home[oof_mask],
            away[oof_mask],
            oof_margin,
            oof_total,
            seasons[oof_mask],
        )

        self.point_model_ = MarginTotalEnsemble(random_state=self.random_state).fit(X, home, away)
        raw_pred_margin, raw_pred_total = self.point_model_.predict(X)
        pred_margin = self.margin_calibrator_.predict(raw_pred_margin)
        pred_total = self.total_calibrator_.predict(raw_pred_total)
        self.training_metrics_ = {
            "games": float(len(train)),
            "margin_mae_in_sample": float(mean_absolute_error(actual_margin, pred_margin)),
            "margin_rmse_in_sample": float(mean_squared_error(actual_margin, pred_margin) ** 0.5),
            "total_mae_in_sample": float(mean_absolute_error(actual_total, pred_total)),
            "total_rmse_in_sample": float(mean_squared_error(actual_total, pred_total) ** 0.5),
            "oof_games_for_distribution": float(oof_mask.sum()),
            "margin_mae_oof_raw": float(
                mean_absolute_error(actual_margin[oof_mask], raw_oof_margin[oof_mask])
            ),
            "margin_mae_oof_calibrated": float(
                mean_absolute_error(actual_margin[oof_mask], oof_margin)
            ),
            "total_mae_oof_raw": float(
                mean_absolute_error(actual_total[oof_mask], raw_oof_total[oof_mask])
            ),
            "total_mae_oof_calibrated": float(
                mean_absolute_error(actual_total[oof_mask], oof_total)
            ),
            "margin_calibration_scale": float(self.margin_calibrator_.scale),
            "margin_calibration_offset": float(self.margin_calibrator_.offset),
            "margin_calibration_prior_strength": margin_prior_strength,
            "margin_calibration_validation_mae": margin_calibration_validation_mae,
            "total_calibration_scale": float(self.total_calibrator_.scale),
            "total_calibration_offset": float(self.total_calibrator_.offset),
            "total_calibration_prior_strength": total_prior_strength,
            "total_calibration_validation_mae": total_calibration_validation_mae,
        }
        return self

    def _expanding_oof(
        self, X: np.ndarray, home: np.ndarray, away: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(home)
        min_train = max(80, int(n * 0.40))
        boundaries = np.linspace(min_train, n, self.oof_folds + 1, dtype=int)
        oof_margin = np.full(n, np.nan)
        oof_total = np.full(n, np.nan)
        for fold in range(self.oof_folds):
            start, end = boundaries[fold], boundaries[fold + 1]
            if end <= start or start < 30:
                continue
            model = MarginTotalEnsemble(random_state=self.random_state + 1000 + fold)
            model.fit(X[:start], home[:start], away[:start])
            margin, total = model.predict(X[start:end])
            oof_margin[start:end], oof_total[start:end] = margin, total
        mask = np.isfinite(oof_margin) & np.isfinite(oof_total)
        return oof_margin, oof_total, mask

    def predict_point(self, row: pd.Series | pd.DataFrame) -> PointPrediction:
        if self.point_model_ is None:
            raise RuntimeError("model is not fitted")
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        X = frame.reindex(columns=self.feature_columns_).astype(float).to_numpy()
        raw_margin, raw_total = self.point_model_.predict(X)
        # getattr keeps artifacts saved before the calibration layer backward compatible.
        margin_calibrator = getattr(self, "margin_calibrator_", AffineTargetCalibrator())
        total_calibrator = getattr(self, "total_calibrator_", AffineTargetCalibrator())
        margin = margin_calibrator.predict(raw_margin)
        total = total_calibrator.predict(raw_total)
        return PointPrediction(float(margin[0]), float(total[0]))

    def predict_distribution(self, row: pd.Series | pd.DataFrame) -> JointScoreDistribution:
        if self.distribution_ is None:
            raise RuntimeError("model is not fitted")
        point = self.predict_point(row)
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        season = None
        if "season" in frame.columns and pd.notna(frame.iloc[0]["season"]):
            season = int(frame.iloc[0]["season"])
        return self.distribution_.predict(point.predicted_margin, point.predicted_total, season=season)

    def price_game(
        self,
        row: pd.Series | pd.DataFrame,
        home_spread: float | None = None,
        total: float | None = None,
    ) -> dict:
        dist = self.predict_distribution(row)
        point = self.predict_point(row)
        output: dict = {
            "point": {
                "home_score": point.predicted_home_score,
                "away_score": point.predicted_away_score,
                "margin": point.predicted_margin,
                "total": point.predicted_total,
            },
            "moneyline": {
                "home": dist.moneyline("home").as_dict(),
                "away": dist.moneyline("away").as_dict(),
            },
        }
        if home_spread is not None:
            output["spread"] = {
                "line": float(home_spread),
                "home": dist.spread(home_spread, "home").as_dict(),
                "away": dist.spread(home_spread, "away").as_dict(),
            }
        if total is not None:
            output["total_market"] = {
                "line": float(total),
                "over": dist.total(total, "over").as_dict(),
                "under": dist.total(total, "under").as_dict(),
            }
        return output

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> NFLPredictor:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("artifact is not an NFLPredictor")
        return obj
