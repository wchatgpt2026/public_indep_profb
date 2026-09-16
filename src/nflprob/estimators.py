from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _ridge(random_state: int) -> Pipeline:
    del random_state
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=12.0)),
        ]
    )


def _hgb(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.035,
                    max_iter=350,
                    max_leaf_nodes=15,
                    min_samples_leaf=18,
                    l2_regularization=3.0,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _extra_trees(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=350,
                    min_samples_leaf=4,
                    max_features=0.85,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


@dataclass
class BlendDiagnostics:
    weights: np.ndarray
    validation_rmse: float


class EnsembleRegressor:
    """Chronologically blended ridge + gradient boosting + extra trees."""

    def __init__(self, random_state: int = 7) -> None:
        self.random_state = int(random_state)
        self.models_: list = []
        self.weights_: np.ndarray | None = None
        self.diagnostics_: BlendDiagnostics | None = None

    def _make_models(self) -> list:
        return [
            _ridge(self.random_state),
            _hgb(self.random_state + 1),
            _extra_trees(self.random_state + 2),
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnsembleRegressor:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < 30:
            raise ValueError("at least 30 rows are required")

        split = max(20, int(n * 0.80))
        split = min(split, n - 10)
        provisional = self._make_models()
        val_predictions: list[np.ndarray] = []
        for model in provisional:
            model.fit(X[:split], y[:split])
            val_predictions.append(model.predict(X[split:]))
        P = np.column_stack(val_predictions)
        target = y[split:]

        def loss(w: np.ndarray) -> float:
            residual = target - P @ w
            return float(np.mean(residual * residual))

        initial = np.full(P.shape[1], 1.0 / P.shape[1])
        result = minimize(
            loss,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * P.shape[1],
            constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
            options={"maxiter": 200, "ftol": 1e-10},
        )
        weights = result.x if result.success else initial
        weights = np.maximum(weights, 0.0)
        weights = weights / weights.sum()

        self.models_ = self._make_models()
        for model in self.models_:
            model.fit(X, y)
        self.weights_ = weights
        self.diagnostics_ = BlendDiagnostics(weights=weights, validation_rmse=loss(weights) ** 0.5)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models_ or self.weights_ is None:
            raise RuntimeError("model is not fitted")
        preds = np.column_stack(
            [model.predict(np.asarray(X, dtype=float)) for model in self.models_]
        )
        return preds @ self.weights_


class MarginTotalEnsemble:
    """Separate target ensembles for margin and total."""

    def __init__(self, random_state: int = 7) -> None:
        self.margin_model = EnsembleRegressor(random_state=random_state)
        self.total_model = EnsembleRegressor(random_state=random_state + 101)

    def fit(
        self,
        X: np.ndarray,
        home_score: np.ndarray,
        away_score: np.ndarray,
    ) -> MarginTotalEnsemble:
        home_score = np.asarray(home_score, dtype=float)
        away_score = np.asarray(away_score, dtype=float)
        self.margin_model.fit(X, home_score - away_score)
        self.total_model.fit(X, home_score + away_score)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        margin = self.margin_model.predict(X)
        total = self.total_model.predict(X)
        return margin, total
