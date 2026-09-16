from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


PBP_METRICS = [
    "off_epa_per_play", "off_success_rate", "off_dropback_epa", "off_rush_epa",
    "off_explosive_rate", "off_turnover_rate", "off_sack_rate", "off_cpoe",
]
ROLLING_METRICS = PBP_METRICS + [
    "def_epa_allowed", "def_success_allowed", "def_dropback_epa_allowed",
    "def_rush_epa_allowed", "def_explosive_allowed", "def_takeaway_rate",
    "def_sack_rate", "points_for", "points_against",
]
PRIORS = {
    "off_epa_per_play": 0.0, "off_success_rate": 0.43, "off_dropback_epa": 0.0,
    "off_rush_epa": -0.02, "off_explosive_rate": 0.11, "off_turnover_rate": 0.018,
    "off_sack_rate": 0.065, "off_cpoe": 0.0, "def_epa_allowed": 0.0,
    "def_success_allowed": 0.43, "def_dropback_epa_allowed": 0.0,
    "def_rush_epa_allowed": -0.02, "def_explosive_allowed": 0.11,
    "def_takeaway_rate": 0.018, "def_sack_rate": 0.065,
    "points_for": 22.0, "points_against": 22.0,
}


def _series(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def aggregate_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate nflverse PBP to one offensive row per team-game."""
    required = {"game_id", "posteam", "defteam"}
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing required columns: {sorted(missing)}")
    epa = _series(pbp, "epa")
    play = _series(pbp, "play", 1.0).fillna(0.0)
    qb_dropback = _series(pbp, "qb_dropback", 0.0).fillna(0.0)
    rush = _series(pbp, "rush", 0.0).fillna(0.0)
    success = _series(pbp, "success")
    yards = _series(pbp, "yards_gained")
    interception = _series(pbp, "interception", 0.0).fillna(0.0)
    fumble_lost = _series(pbp, "fumble_lost", 0.0).fillna(0.0)
    sack = _series(pbp, "sack", 0.0).fillna(0.0)
    cpoe = _series(pbp, "cpoe")
    mask = (play == 1) & pbp["posteam"].notna() & pbp["defteam"].notna() & epa.notna()
    work = pbp.loc[mask, ["game_id", "posteam", "defteam"]].copy()
    work["epa"] = epa.loc[mask]
    work["success"] = success.loc[mask]
    work["dropback"] = qb_dropback.loc[mask]
    work["rush"] = rush.loc[mask]
    work["explosive"] = (yards.loc[mask] >= 20.0).astype(float)
    work["turnover"] = ((interception + fumble_lost).loc[mask] > 0).astype(float)
    work["sack"] = sack.loc[mask]
    work["cpoe"] = cpoe.loc[mask]

    def summarize(group: pd.DataFrame) -> pd.Series:
        dropbacks = group["dropback"] > 0
        rushes = group["rush"] > 0
        return pd.Series({
            "off_epa_per_play": group["epa"].mean(),
            "off_success_rate": group["success"].mean(),
            "off_dropback_epa": group.loc[dropbacks, "epa"].mean(),
            "off_rush_epa": group.loc[rushes, "epa"].mean(),
            "off_explosive_rate": group["explosive"].mean(),
            "off_turnover_rate": group["turnover"].mean(),
            "off_sack_rate": group.loc[dropbacks, "sack"].mean(),
            "off_cpoe": group.loc[dropbacks, "cpoe"].mean(),
            "plays": len(group),
        })
    out = work.groupby(["game_id", "posteam", "defteam"], observed=True).apply(
        summarize, include_groups=False
    )
    return out.reset_index().rename(columns={"posteam": "team", "defteam": "opponent"})


def aggregate_qb_pbp(pbp: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe quarterback state histories from prior dropbacks."""
    passer_col = next((c for c in ("passer_player_id", "passer_id") if c in pbp.columns), None)
    columns = ["qb_id", "_date", "qb_epa_state", "qb_cpoe_state", "qb_games"]
    if passer_col is None:
        return pd.DataFrame(columns=columns)
    epa = _series(pbp, "epa")
    dropback = _series(pbp, "qb_dropback", 0.0).fillna(0.0)
    cpoe = _series(pbp, "cpoe")
    mask = (dropback > 0) & pbp[passer_col].notna() & epa.notna()
    if not mask.any():
        return pd.DataFrame(columns=columns)
    work = pd.DataFrame({
        "game_id": pbp.loc[mask, "game_id"].astype(str),
        "qb_id": pbp.loc[mask, passer_col].astype(str),
        "epa": epa.loc[mask].to_numpy(),
        "cpoe": cpoe.loc[mask].to_numpy(),
    })
    game_qb = (
        work.groupby(["game_id", "qb_id"], observed=True)
        .agg(qb_epa=("epa", "mean"), qb_cpoe=("cpoe", "mean"), dropbacks=("epa", "size"))
        .reset_index()
    )
    dates = schedule[["game_id", "_date"]].copy()
    dates["game_id"] = dates["game_id"].astype(str)
    game_qb = game_qb.merge(dates, on="game_id", how="left")
    game_qb = game_qb.loc[game_qb["_date"].notna()].copy()
    game_qb = game_qb.sort_values(["qb_id", "_date", "game_id"], kind="stable")
    alpha = 1.0 - math.exp(math.log(0.5) / 8.0)
    game_qb["qb_epa_state"] = game_qb.groupby("qb_id", observed=True)["qb_epa"].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    )
    game_qb["qb_cpoe_state"] = game_qb.groupby("qb_id", observed=True)["qb_cpoe"].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    )
    game_qb["qb_games"] = game_qb.groupby("qb_id", observed=True).cumcount() + 1
    return game_qb[columns]


def _add_qb_features(games: pd.DataFrame, schedule: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    history = aggregate_qb_pbp(pbp, schedule)
    lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for qb_id, group in history.groupby("qb_id", observed=True):
        ordered = group.sort_values("_date", kind="stable")
        lookup[str(qb_id)] = (
            ordered["_date"].to_numpy(dtype="datetime64[ns]"),
            ordered["qb_epa_state"].to_numpy(dtype=float),
            ordered["qb_cpoe_state"].to_numpy(dtype=float),
            ordered["qb_games"].to_numpy(dtype=float),
        )
    for side in ("home", "away"):
        qb_col = f"{side}_qb_id"
        qb_series = schedule[qb_col] if qb_col in schedule.columns else pd.Series(None, index=schedule.index)
        epa_values, cpoe_values, experience_values, known_values = [], [], [], []
        for qb_raw, game_date in zip(qb_series, schedule["_date"], strict=True):
            qb_id = str(qb_raw) if pd.notna(qb_raw) else None
            state = lookup.get(qb_id) if qb_id is not None else None
            if state is None or pd.isna(game_date):
                epa_values.append(0.0); cpoe_values.append(0.0)
                experience_values.append(0.0); known_values.append(0.0)
                continue
            dates, epa_state, cpoe_state, qb_games = state
            idx = int(np.searchsorted(dates, np.datetime64(game_date), side="left") - 1)
            if idx < 0:
                epa_values.append(0.0); cpoe_values.append(0.0)
                experience_values.append(0.0); known_values.append(1.0)
                continue
            seen = float(qb_games[idx])
            shrink = seen / (seen + 3.0)
            epa_values.append(float(epa_state[idx]) * shrink)
            cpoe_values.append(float(np.nan_to_num(cpoe_state[idx], nan=0.0)) * shrink)
            experience_values.append(min(seen, 32.0) / 32.0)
            known_values.append(1.0)
        games[f"{side}_qb_epa"] = epa_values
        games[f"{side}_qb_cpoe"] = cpoe_values
        games[f"{side}_qb_experience"] = experience_values
        games[f"{side}_qb_known"] = known_values
    games["qb_epa_diff"] = games["home_qb_epa"] - games["away_qb_epa"]
    games["qb_cpoe_diff"] = games["home_qb_cpoe"] - games["away_qb_cpoe"]
    games["qb_experience_diff"] = games["home_qb_experience"] - games["away_qb_experience"]
    return games


def _schedule_dates(schedule: pd.DataFrame) -> pd.Series:
    for col in ("gameday", "game_date", "date"):
        if col in schedule.columns:
            return pd.to_datetime(schedule[col], errors="coerce")
    return pd.Series(pd.NaT, index=schedule.index)


def _elo_pregame(schedule: pd.DataFrame, k: float = 22.0, home_advantage: float = 55.0) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    previous_season: int | None = None
    rows: list[dict[str, float | str]] = []
    ordered = schedule.sort_values(["season", "week", "_date", "game_id"], kind="stable")
    for row in ordered.itertuples(index=False):
        season = int(row.season)
        if previous_season is not None and season != previous_season:
            for team in list(ratings):
                ratings[team] = 1500.0 + 0.67 * (ratings[team] - 1500.0)
        previous_season = season
        home, away = str(row.home_team), str(row.away_team)
        neutral = bool(getattr(row, "neutral", False))
        h, a = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        hfa = 0.0 if neutral else home_advantage
        rows.append({"game_id": row.game_id, "home_elo": h, "away_elo": a})
        if pd.notna(row.home_score) and pd.notna(row.away_score):
            expected = 1.0 / (1.0 + 10.0 ** (-(h + hfa - a) / 400.0))
            actual = 1.0 if row.home_score > row.away_score else 0.0 if row.home_score < row.away_score else 0.5
            multiplier = math.log1p(max(abs(float(row.home_score) - float(row.away_score)), 1.0)) * 1.1
            delta = k * multiplier * (actual - expected)
            ratings[home], ratings[away] = h + delta, a - delta
    return pd.DataFrame(rows)


def build_pregame_features(schedule: pd.DataFrame, pbp: pd.DataFrame, half_life_games: float = 6.0) -> pd.DataFrame:
    """Build leakage-safe features for completed and future games."""
    required = {"game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule missing required columns: {sorted(missing)}")
    schedule = schedule.copy()
    schedule["_date"] = _schedule_dates(schedule)
    if "neutral" not in schedule.columns:
        schedule["neutral"] = False
    offense = aggregate_pbp(pbp)
    defensive = offense[["game_id", "team"] + PBP_METRICS].copy().rename(columns={
        "team": "opponent", "off_epa_per_play": "def_epa_allowed",
        "off_success_rate": "def_success_allowed", "off_dropback_epa": "def_dropback_epa_allowed",
        "off_rush_epa": "def_rush_epa_allowed", "off_explosive_rate": "def_explosive_allowed",
        "off_turnover_rate": "def_takeaway_rate", "off_sack_rate": "def_sack_rate",
        "off_cpoe": "_opp_cpoe",
    })

    def side_rows(side: str) -> pd.DataFrame:
        other = "away" if side == "home" else "home"
        cols = ["game_id", "season", "week", "_date", f"{side}_team", f"{other}_team", f"{side}_score", f"{other}_score"]
        out = schedule[cols].copy()
        out.columns = ["game_id", "season", "week", "_date", "team", "opponent", "points_for", "points_against"]
        out["is_home"] = 1.0 if side == "home" else 0.0
        return out

    team_games = pd.concat([side_rows("home"), side_rows("away")], ignore_index=True)
    team_games = team_games.merge(offense.drop(columns=["opponent"]), on=["game_id", "team"], how="left")
    team_games = team_games.merge(defensive, on=["game_id", "opponent"], how="left")
    team_games = team_games.sort_values(["team", "_date", "season", "week", "game_id"], kind="stable")
    alpha = 1.0 - math.exp(math.log(0.5) / float(half_life_games))
    for metric in ROLLING_METRICS:
        if metric not in team_games.columns:
            team_games[metric] = np.nan
        team_games[f"pregame_{metric}"] = team_games.groupby("team", observed=True)[metric].transform(
            lambda x: x.shift(1).ewm(alpha=alpha, adjust=False, min_periods=1).mean()
        ).fillna(PRIORS[metric])
    team_games["rest_days"] = team_games.groupby("team", observed=True)["_date"].diff().dt.days
    team_games["rest_days"] = team_games["rest_days"].clip(lower=4, upper=21).fillna(7.0)
    feature_cols = [f"pregame_{m}" for m in ROLLING_METRICS] + ["rest_days"]
    home = team_games.loc[team_games["is_home"] == 1.0, ["game_id"] + feature_cols].rename(
        columns={c: f"home_{c}" for c in feature_cols}
    )
    away = team_games.loc[team_games["is_home"] == 0.0, ["game_id"] + feature_cols].rename(
        columns={c: f"away_{c}" for c in feature_cols}
    )
    games = schedule.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")
    for metric in ROLLING_METRICS:
        h, a = f"home_pregame_{metric}", f"away_pregame_{metric}"
        games[f"diff_{metric}"] = games[h] - games[a]
        games[f"sum_{metric}"] = games[h] + games[a]
    games["rest_diff"] = games["home_rest_days"] - games["away_rest_days"]
    elo = _elo_pregame(schedule)
    games = games.merge(elo, on="game_id", how="left")
    games["elo_diff"] = games["home_elo"] - games["away_elo"]
    games["elo_sum_centered"] = games["home_elo"] + games["away_elo"] - 3000.0
    games = _add_qb_features(games, schedule, pbp)
    games["neutral_site"] = games["neutral"].fillna(False).astype(float)
    games["week_sin"] = np.sin(2.0 * np.pi * pd.to_numeric(games["week"], errors="coerce") / 22.0)
    games["week_cos"] = np.cos(2.0 * np.pi * pd.to_numeric(games["week"], errors="coerce") / 22.0)
    for raw, engineered in (("temp", "temperature"), ("wind", "wind_speed")):
        games[engineered] = pd.to_numeric(games[raw], errors="coerce") if raw in games.columns else np.nan
    if "roof" in games.columns:
        games["indoors"] = games["roof"].astype(str).str.lower().isin(["closed", "dome"]).astype(float)
    else:
        games["indoors"] = np.nan
    games["game_date"] = games["_date"]
    return games.drop(columns=["_date"])


def numeric_feature_columns(frame: pd.DataFrame, extra_exclude: Iterable[str] = ()) -> list[str]:
    """Choose only engineered pregame numeric inputs by default."""
    blocked = set(extra_exclude) | {"home_score", "away_score", "result", "total", "overtime"}
    market_tokens = ("moneyline", "spread_line", "total_line", "odds", "vegas", "market_")
    exact_context = {
        "home_rest_days", "away_rest_days", "rest_diff", "home_elo", "away_elo",
        "elo_diff", "elo_sum_centered", "neutral_site", "week_sin", "week_cos",
        "temperature", "wind_speed", "indoors", "home_qb_epa", "away_qb_epa",
        "home_qb_cpoe", "away_qb_cpoe", "home_qb_experience", "away_qb_experience",
        "home_qb_known", "away_qb_known", "qb_epa_diff", "qb_cpoe_diff", "qb_experience_diff",
    }
    numeric = set(frame.select_dtypes(include=[np.number, "bool"]).columns)
    columns: list[str] = []
    for col in frame.columns:
        if col not in numeric or col in blocked or any(t in col.lower() for t in market_tokens):
            continue
        if (
            col.startswith("home_pregame_") or col.startswith("away_pregame_")
            or col.startswith("diff_") or col.startswith("sum_") or col in exact_context
        ):
            columns.append(col)
    return columns
