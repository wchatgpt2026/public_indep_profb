from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .features import build_pregame_features


def load_nflverse(seasons: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load nflverse schedule and PBP using the official Python reader."""
    try:
        import nflreadpy as nfl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install data dependencies with: pip install 'nflprob[data]'") from exc

    seasons = [int(s) for s in seasons]
    pbp = nfl.load_pbp(seasons).to_pandas()
    schedule = nfl.load_schedules(seasons).to_pandas()
    return schedule, pbp


def build_nflverse_dataset(
    seasons: Iterable[int],
    half_life_games: float = 6.0,
) -> pd.DataFrame:
    schedule, pbp = load_nflverse(seasons)
    return build_pregame_features(schedule, pbp, half_life_games=half_life_games)
