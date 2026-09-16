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


class NFLPredictor:
    """Independent NFL point model + calibrated full score distribution."""

    def __init__(self, random_state: int = 7, score_max: int = 80, oof_folds: int = 4) -> None:
        self.random_state = int(random_state)
        self.score_max = int(score_max)
        self.oof_folds = int(oof_folds)
        self.feature_columns_: list[str] = []
        self.point_model_: MarginTotalEnsemble | None = None
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
        oof_margin, oof_total, oof_mask = self._expanding_oof(X, home, away)
        if oof_mask.sum() < 50:
            raise ValueError("not enough chronological out-of-fold predictions for calibration")
        seasons = (
            train["season"].to_numpy(dtype=float)
            if "season" in train.columns else np.zeros(len(train), dtype=float)
        )
        self.distribution_ = ScoreDistributionCalibrator(score_max=self.score_max)
        self.distribution_.fit(
            home[oof_mask], away[oof_mask], oof_margin[oof_mask], oof_total[oof_mask], seasons[oof_mask]
        )
        self.point_model_ = MarginTotalEnsemble(random_state=self.random_state).fit(X, home, away)
        pred_margin, pred_total = self.point_model_.predict(X)
        actual_margin, actual_total = home - away, home + away
        self.training_metrics_ = {
            "games": float(len(train)),
            "margin_mae_in_sample": float(mean_absolute_error(actual_margin, pred_margin)),
            "margin_rmse_in_sample": float(mean_squared_error(actual_margin, pred_margin) ** 0.5),
            "total_mae_in_sample": float(mean_absolute_error(actual_total, pred_total)),
            "total_rmse_in_sample": float(mean_squared_error(actual_total, pred_total) ** 0.5),
            "oof_games_for_distribution": float(oof_mask.sum()),
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
        margin, total = self.point_model_.predict(X)
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
