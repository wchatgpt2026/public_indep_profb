from __future__ import annotations

import numpy as np
import pandas as pd

from .features import numeric_feature_columns
from .model import is_experimental_feature


MARKET_COLUMNS = (
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
)


def _coverage(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.notna().mean())


def _numeric_nonzero_rate(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or len(frame) == 0:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0
    return float((finite.abs() > 1e-12).mean())


def _qb_summary(frame: pd.DataFrame) -> dict[str, object]:
    required = ["home_qb_known", "away_qb_known"]
    present = {column: column in frame.columns for column in required}
    summary: dict[str, object] = {"columns_present": present}
    if not all(present.values()) or frame.empty:
        return summary

    home_known = pd.to_numeric(frame["home_qb_known"], errors="coerce").fillna(0.0) > 0.5
    away_known = pd.to_numeric(frame["away_qb_known"], errors="coerce").fillna(0.0) > 0.5
    summary.update(
        {
            "home_known_rate": float(home_known.mean()),
            "away_known_rate": float(away_known.mean()),
            "both_known_rate": float((home_known & away_known).mean()),
            "either_unknown_rate": float((~(home_known & away_known)).mean()),
            "home_qb_epa_nonzero_rate": _numeric_nonzero_rate(frame, "home_qb_epa"),
            "away_qb_epa_nonzero_rate": _numeric_nonzero_rate(frame, "away_qb_epa"),
            "home_qb_experience_nonzero_rate": _numeric_nonzero_rate(
                frame, "home_qb_experience"
            ),
            "away_qb_experience_nonzero_rate": _numeric_nonzero_rate(
                frame, "away_qb_experience"
            ),
        }
    )
    if "season" in frame.columns:
        by_season: list[dict[str, float | int]] = []
        seasons = pd.to_numeric(frame["season"], errors="coerce")
        for season in sorted(int(value) for value in seasons.dropna().unique()):
            mask = seasons == season
            h = home_known.loc[mask]
            a = away_known.loc[mask]
            by_season.append(
                {
                    "season": season,
                    "games": int(mask.sum()),
                    "home_known_rate": float(h.mean()),
                    "away_known_rate": float(a.mean()),
                    "both_known_rate": float((h & a).mean()),
                }
            )
        summary["by_season"] = by_season
    return summary


def _weather_summary(frame: pd.DataFrame) -> dict[str, object]:
    columns = ("temperature", "wind_speed", "indoors")
    return {
        column: {
            "present": column in frame.columns,
            "coverage": _coverage(frame[column]) if column in frame.columns else 0.0,
        }
        for column in columns
    }


def _experimental_summary(frame: pd.DataFrame) -> dict[str, object]:
    all_features = numeric_feature_columns(frame, extra_exclude={"season", "week"})
    experimental = [column for column in all_features if is_experimental_feature(column)]
    coverage = {
        column: _coverage(frame[column])
        for column in experimental
        if column in frame.columns
    }
    return {
        "count": len(experimental),
        "min_coverage": float(min(coverage.values())) if coverage else 0.0,
        "median_coverage": float(np.median(list(coverage.values()))) if coverage else 0.0,
        "max_coverage": float(max(coverage.values())) if coverage else 0.0,
        "lowest_coverage_features": [
            {"feature": name, "coverage": float(value)}
            for name, value in sorted(coverage.items(), key=lambda item: item[1])[:10]
        ],
    }


def _market_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        column: {
            "present": column in frame.columns,
            "coverage": _coverage(frame[column]) if column in frame.columns else 0.0,
        }
        for column in MARKET_COLUMNS
    }


def audit_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize modeling-input coverage without fitting a model."""
    rows = len(frame)
    completed = (
        int((frame["home_score"].notna() & frame["away_score"].notna()).sum())
        if {"home_score", "away_score"}.issubset(frame.columns)
        else 0
    )
    seasons = (
        pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int)
        if "season" in frame.columns
        else pd.Series(dtype=int)
    )
    accepted_features = [
        column
        for column in numeric_feature_columns(frame, extra_exclude={"season", "week"})
        if not is_experimental_feature(column)
    ]
    return {
        "rows": int(rows),
        "completed_games": completed,
        "columns": int(len(frame.columns)),
        "season_min": int(seasons.min()) if len(seasons) else None,
        "season_max": int(seasons.max()) if len(seasons) else None,
        "accepted_feature_count": len(accepted_features),
        "qb": _qb_summary(frame),
        "weather": _weather_summary(frame),
        "experimental_features": _experimental_summary(frame),
        "market_fields_evaluation_only": _market_summary(frame),
    }
