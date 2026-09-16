from __future__ import annotations

import numpy as np
import pandas as pd

from .distribution import JointScoreDistribution, ScoreDistributionCalibrator
from .estimators import EnsembleRegressor
from .model import NFLPredictor, PointPrediction


class DevelopmentTargetSplitPredictor:
    """Research-only predictor with separate feature matrices for margin and total.

    This class exists to test target-specific feature augmentation without changing the
    production NFLPredictor defaults. Margin and total models remain statistically
    independent; their chronological OOF predictions are then combined only in the score
    distribution calibrator.
    """

    def __init__(self, random_state: int = 7, score_max: int = 80, oof_folds: int = 4) -> None:
        self.random_state = int(random_state)
        self.score_max = int(score_max)
        self.oof_folds = int(oof_folds)
        self.margin_feature_columns_: list[str] = []
        self.total_feature_columns_: list[str] = []
        self.margin_model_: EnsembleRegressor | None = None
        self.total_model_: EnsembleRegressor | None = None
        self.distribution_: ScoreDistributionCalibrator | None = None

    def fit(
        self,
        games: pd.DataFrame,
        *,
        margin_feature_columns: list[str],
        total_feature_columns: list[str],
    ) -> DevelopmentTargetSplitPredictor:
        required = {"home_score", "away_score"}
        missing = required - set(games.columns)
        if missing:
            raise ValueError(f"training data missing targets: {sorted(missing)}")
        NFLPredictor._assert_independent(margin_feature_columns)
        NFLPredictor._assert_independent(total_feature_columns)
        if not margin_feature_columns or not total_feature_columns:
            raise ValueError("margin and total feature sets must both be non-empty")

        train = games.loc[games["home_score"].notna() & games["away_score"].notna()].copy()
        train = NFLPredictor._sort_games(train)
        if len(train) < 120:
            raise ValueError("at least 120 completed games are required for fitting")

        self.margin_feature_columns_ = list(margin_feature_columns)
        self.total_feature_columns_ = list(total_feature_columns)
        X_margin = train[self.margin_feature_columns_].astype(float).to_numpy()
        X_total = train[self.total_feature_columns_].astype(float).to_numpy()
        home = train["home_score"].astype(float).to_numpy()
        away = train["away_score"].astype(float).to_numpy()

        oof_margin, oof_total, oof_mask = self._expanding_oof(
            X_margin,
            X_total,
            home,
            away,
        )
        if oof_mask.sum() < 50:
            raise ValueError("not enough chronological out-of-fold predictions for calibration")
        seasons = (
            train["season"].to_numpy(dtype=float)
            if "season" in train.columns
            else np.zeros(len(train), dtype=float)
        )
        self.distribution_ = ScoreDistributionCalibrator(score_max=self.score_max)
        self.distribution_.fit(
            home[oof_mask],
            away[oof_mask],
            oof_margin[oof_mask],
            oof_total[oof_mask],
            seasons[oof_mask],
        )

        self.margin_model_ = EnsembleRegressor(random_state=self.random_state).fit(
            X_margin,
            home - away,
        )
        self.total_model_ = EnsembleRegressor(random_state=self.random_state + 101).fit(
            X_total,
            home + away,
        )
        return self

    def _expanding_oof(
        self,
        X_margin: np.ndarray,
        X_total: np.ndarray,
        home: np.ndarray,
        away: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(home)
        min_train = max(80, int(n * 0.40))
        boundaries = np.linspace(min_train, n, self.oof_folds + 1, dtype=int)
        oof_margin = np.full(n, np.nan)
        oof_total = np.full(n, np.nan)
        margin_target = home - away
        total_target = home + away
        for fold in range(self.oof_folds):
            start, end = boundaries[fold], boundaries[fold + 1]
            if end <= start or start < 30:
                continue
            margin_model = EnsembleRegressor(random_state=self.random_state + 1000 + fold)
            total_model = EnsembleRegressor(random_state=self.random_state + 1101 + fold)
            margin_model.fit(X_margin[:start], margin_target[:start])
            total_model.fit(X_total[:start], total_target[:start])
            oof_margin[start:end] = margin_model.predict(X_margin[start:end])
            oof_total[start:end] = total_model.predict(X_total[start:end])
        mask = np.isfinite(oof_margin) & np.isfinite(oof_total)
        return oof_margin, oof_total, mask

    def predict_point(self, row: pd.Series | pd.DataFrame) -> PointPrediction:
        if self.margin_model_ is None or self.total_model_ is None:
            raise RuntimeError("model is not fitted")
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        X_margin = frame.reindex(columns=self.margin_feature_columns_).astype(float).to_numpy()
        X_total = frame.reindex(columns=self.total_feature_columns_).astype(float).to_numpy()
        margin = self.margin_model_.predict(X_margin)
        total = self.total_model_.predict(X_total)
        return PointPrediction(float(margin[0]), float(total[0]))

    def predict_distribution(self, row: pd.Series | pd.DataFrame) -> JointScoreDistribution:
        if self.distribution_ is None:
            raise RuntimeError("model is not fitted")
        point = self.predict_point(row)
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        season = None
        if "season" in frame.columns and pd.notna(frame.iloc[0]["season"]):
            season = int(frame.iloc[0]["season"])
        return self.distribution_.predict(
            point.predicted_margin,
            point.predicted_total,
            season=season,
        )
