from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


INJURY_LAST_AVAILABLE_SEASON = 2024
EASTERN_TIMEZONE = "America/New_York"


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    raise TypeError("nflverse loader did not return a pandas- or Polars-compatible frame")


def _load_with_retries(loader: Any, season: int, attempts: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _to_pandas(loader(season))
        except (ConnectionError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4))
    raise ConnectionError(f"Failed to load nflverse context for {season}") from last_error


def load_pregame_context_sources(
    seasons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load schedules, injury reports, and depth charts for a context audit.

    Injury data is intentionally requested only through 2024 because the upstream nflverse
    injury source stopped after that season. Depth charts are loaded one season at a time so
    the pre-2025 and 2025+ schemas can coexist in one pandas frame.
    """
    try:
        import nflreadpy as nfl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install data dependencies with: pip install 'nflprob[data]'") from exc

    requested = sorted({int(season) for season in seasons})
    if not requested:
        raise ValueError("at least one season is required")

    schedules = _to_pandas(nfl.load_schedules(requested))

    injury_frames: list[pd.DataFrame] = []
    for season in requested:
        if season > INJURY_LAST_AVAILABLE_SEASON:
            continue
        frame = _load_with_retries(nfl.load_injuries, season)
        if "season" not in frame.columns:
            frame["season"] = season
        injury_frames.append(frame)
    injuries = (
        pd.concat(injury_frames, ignore_index=True, sort=False)
        if injury_frames
        else pd.DataFrame()
    )

    depth_frames: list[pd.DataFrame] = []
    for season in requested:
        frame = _load_with_retries(nfl.load_depth_charts, season)
        frame["_requested_season"] = season
        depth_frames.append(frame)
    depth_charts = pd.concat(depth_frames, ignore_index=True, sort=False)
    return schedules, injuries, depth_charts


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _parse_kickoff_utc(schedules: pd.DataFrame) -> pd.Series:
    if not {"gameday", "gametime"}.issubset(schedules.columns):
        return pd.Series(pd.NaT, index=schedules.index, dtype="datetime64[ns, UTC]")
    text = schedules["gameday"].astype("string") + " " + schedules["gametime"].astype("string")
    local = pd.to_datetime(text, errors="coerce")
    localized = local.dt.tz_localize(
        EASTERN_TIMEZONE,
        ambiguous="NaT",
        nonexistent="shift_forward",
    )
    return localized.dt.tz_convert("UTC")


def _team_games(
    schedules: pd.DataFrame,
    *,
    cutoff_hours: float,
    as_of_utc: pd.Timestamp,
) -> pd.DataFrame:
    required = {"season", "week", "home_team", "away_team"}
    missing = required - set(schedules.columns)
    if missing:
        raise ValueError(f"schedule data missing columns: {sorted(missing)}")

    games = schedules.copy()
    games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    games["week"] = pd.to_numeric(games["week"], errors="coerce").astype("Int64")
    games["kickoff_utc"] = _parse_kickoff_utc(games)
    games["cutoff_utc"] = games["kickoff_utc"] - pd.to_timedelta(cutoff_hours, unit="h")
    games["cutoff_matured"] = games["cutoff_utc"].notna() & (games["cutoff_utc"] <= as_of_utc)

    id_columns = [column for column in ("game_id", "game_type") if column in games.columns]
    common = ["season", "week", "kickoff_utc", "cutoff_utc", "cutoff_matured", *id_columns]
    home = games[common + ["home_team"]].rename(columns={"home_team": "team"})
    home["side"] = "home"
    away = games[common + ["away_team"]].rename(columns={"away_team": "team"})
    away["side"] = "away"
    team_games = pd.concat([home, away], ignore_index=True, sort=False)
    team_games["team"] = team_games["team"].astype("string")
    return team_games


def _by_season_rates(
    frame: pd.DataFrame,
    *,
    season_column: str,
    value_columns: tuple[str, ...],
) -> list[dict[str, float | int]]:
    if frame.empty or season_column not in frame.columns:
        return []
    output: list[dict[str, float | int]] = []
    seasons = pd.to_numeric(frame[season_column], errors="coerce")
    for season in sorted(int(value) for value in seasons.dropna().unique()):
        subset = frame.loc[seasons == season]
        row: dict[str, float | int] = {"season": season, "team_games": int(len(subset))}
        for column in value_columns:
            if column in subset.columns:
                row[f"{column}_rate"] = float(subset[column].fillna(False).astype(bool).mean())
        output.append(row)
    return output


def _injury_audit(
    injuries: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    requested_seasons: list[int],
) -> dict[str, object]:
    unavailable = [season for season in requested_seasons if season > INJURY_LAST_AVAILABLE_SEASON]
    summary: dict[str, object] = {
        "upstream_last_available_season": INJURY_LAST_AVAILABLE_SEASON,
        "requested_seasons_without_upstream_data": unavailable,
        "rows": int(len(injuries)),
        "columns": sorted(str(column) for column in injuries.columns),
    }
    if injuries.empty:
        summary["strict_cutoff_ready"] = False
        return summary

    frame = injuries.copy()
    if not {"season", "week", "team"}.issubset(frame.columns):
        summary["strict_cutoff_ready"] = False
        summary["error"] = "injury data lacks season/week/team join keys"
        return summary

    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
    frame["team"] = frame["team"].astype("string")
    modified = (
        pd.to_datetime(frame["date_modified"], errors="coerce", utc=True)
        if "date_modified" in frame.columns
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    )
    frame["modified_utc"] = modified
    summary["date_modified_present"] = "date_modified" in frame.columns
    summary["date_modified_parse_rate"] = float(modified.notna().mean())

    matured = team_games.loc[
        team_games["cutoff_matured"]
        & pd.to_numeric(team_games["season"], errors="coerce").le(INJURY_LAST_AVAILABLE_SEASON)
    ].copy()
    keys = ["season", "week", "team"]
    records = frame.merge(matured[keys + ["cutoff_utc"]], on=keys, how="inner")
    records["eligible_at_cutoff"] = (
        records["modified_utc"].notna() & (records["modified_utc"] <= records["cutoff_utc"])
    )

    any_report = records[keys].drop_duplicates().assign(any_report=True)
    eligible_report = (
        records.loc[records["eligible_at_cutoff"], keys]
        .drop_duplicates()
        .assign(eligible_report=True)
    )
    joined = matured.merge(any_report, on=keys, how="left").merge(
        eligible_report, on=keys, how="left"
    )
    joined["any_report"] = joined["any_report"].fillna(False)
    joined["eligible_report"] = joined["eligible_report"].fillna(False)

    player_key = "gsis_id" if "gsis_id" in frame.columns else None
    revision_groups = 0
    if player_key is not None:
        revision_groups = int(
            (frame.groupby(keys + [player_key], dropna=False).size() > 1).sum()
        )

    qb_rows = frame.loc[
        frame.get("position", pd.Series(index=frame.index, dtype="string"))
        .astype("string")
        .str.upper()
        .eq("QB")
    ]
    summary.update(
        {
            "seasons_present": sorted(
                int(value) for value in frame["season"].dropna().unique()
            ),
            "teams_present": int(frame["team"].nunique(dropna=True)),
            "players_present": int(frame[player_key].nunique(dropna=True)) if player_key else None,
            "qb_rows": int(len(qb_rows)),
            "revision_player_week_groups": revision_groups,
            "matured_team_games": int(len(matured)),
            "team_game_any_report_rate": float(joined["any_report"].mean()) if len(joined) else float("nan"),
            "team_game_report_available_at_cutoff_rate": float(joined["eligible_report"].mean()) if len(joined) else float("nan"),
            "by_season": _by_season_rates(
                joined,
                season_column="season",
                value_columns=("any_report", "eligible_report"),
            ),
            "strict_cutoff_ready": bool(
                len(matured)
                and modified.notna().mean() >= 0.95
                and joined["eligible_report"].mean() > 0.0
            ),
        }
    )
    for column in ("report_status", "practice_status"):
        summary[f"{column}_coverage"] = (
            float(frame[column].notna().mean()) if column in frame.columns else 0.0
        )
    return summary


def _pre2025_depth_audit(depth: pd.DataFrame, team_games: pd.DataFrame) -> dict[str, object]:
    historical = depth.loc[
        pd.to_numeric(depth["_requested_season"], errors="coerce").le(2024)
    ].copy()
    matured = team_games.loc[
        team_games["cutoff_matured"]
        & pd.to_numeric(team_games["season"], errors="coerce").le(2024)
    ].copy()
    summary: dict[str, object] = {
        "rows": int(len(historical)),
        "matured_team_games": int(len(matured)),
        "timestamped": False,
        "strict_cutoff_enforceable": False,
    }
    if historical.empty:
        return summary

    team_column = "club_code" if "club_code" in historical.columns else "team"
    if team_column not in historical.columns or "week" not in historical.columns:
        summary["error"] = "pre-2025 depth chart data lacks team/week join keys"
        return summary

    historical["season_key"] = pd.to_numeric(
        historical.get("season", historical["_requested_season"]), errors="coerce"
    ).astype("Int64")
    historical["week_key"] = pd.to_numeric(historical["week"], errors="coerce").astype("Int64")
    historical["team_key"] = historical[team_column].astype("string")
    qb_mask = (
        historical.get("position", pd.Series(index=historical.index, dtype="string"))
        .astype("string")
        .str.upper()
        .eq("QB")
    )
    qbs = historical.loc[qb_mask].copy()
    if "depth_team" in qbs.columns:
        rank = pd.to_numeric(qbs["depth_team"], errors="coerce")
        qbs = qbs.loc[rank.eq(1)]

    available = (
        qbs[["season_key", "week_key", "team_key"]]
        .drop_duplicates()
        .assign(qb1_weekly_available=True)
    )
    joined = matured.copy()
    joined["season_key"] = pd.to_numeric(joined["season"], errors="coerce").astype("Int64")
    joined["week_key"] = pd.to_numeric(joined["week"], errors="coerce").astype("Int64")
    joined["team_key"] = joined["team"].astype("string")
    joined = joined.merge(available, on=["season_key", "week_key", "team_key"], how="left")
    joined["qb1_weekly_available"] = joined["qb1_weekly_available"].fillna(False)
    summary.update(
        {
            "qb1_weekly_team_game_coverage": float(joined["qb1_weekly_available"].mean()) if len(joined) else float("nan"),
            "by_season": _by_season_rates(
                joined,
                season_column="season",
                value_columns=("qb1_weekly_available",),
            ),
            "interpretation": "joinable by season/week/team, but no record timestamp exists to prove the snapshot was available by a fixed pregame cutoff",
        }
    )
    return summary


def _post2025_depth_audit(depth: pd.DataFrame, team_games: pd.DataFrame) -> dict[str, object]:
    recent = depth.loc[
        pd.to_numeric(depth["_requested_season"], errors="coerce").ge(2025)
    ].copy()
    matured = team_games.loc[
        team_games["cutoff_matured"]
        & pd.to_numeric(team_games["season"], errors="coerce").ge(2025)
    ].copy()
    summary: dict[str, object] = {
        "rows": int(len(recent)),
        "matured_team_games": int(len(matured)),
        "timestamped": "dt" in recent.columns,
    }
    if recent.empty or "dt" not in recent.columns:
        summary["strict_cutoff_enforceable"] = False
        return summary

    recent["snapshot_utc"] = pd.to_datetime(recent["dt"], errors="coerce", utc=True)
    summary["timestamp_parse_rate"] = float(recent["snapshot_utc"].notna().mean())
    required = {"team", "pos_abb"}
    if not required.issubset(recent.columns):
        summary["strict_cutoff_enforceable"] = False
        summary["error"] = "2025+ depth chart data lacks team/pos_abb fields"
        return summary

    recent["season_key"] = pd.to_numeric(recent["_requested_season"], errors="coerce").astype("Int64")
    recent["team_key"] = recent["team"].astype("string")
    qbs = recent.loc[recent["pos_abb"].astype("string").str.upper().eq("QB")].copy()
    if "pos_rank" in qbs.columns:
        qbs["rank"] = pd.to_numeric(qbs["pos_rank"], errors="coerce")
        minimum_rank = qbs.groupby(["season_key", "team_key", "snapshot_utc"])["rank"].transform("min")
        qbs = qbs.loc[qbs["rank"].eq(minimum_rank)]
    qbs = qbs.sort_values(["season_key", "team_key", "snapshot_utc"], kind="stable")
    qbs = qbs.drop_duplicates(["season_key", "team_key", "snapshot_utc"], keep="first")

    results: list[pd.DataFrame] = []
    for (season, team), games_group in matured.groupby(["season", "team"], dropna=False):
        snapshots = qbs.loc[
            qbs["season_key"].eq(int(season)) & qbs["team_key"].eq(str(team))
        ].copy()
        left = games_group.sort_values("cutoff_utc").copy()
        if snapshots.empty:
            left["snapshot_utc"] = pd.NaT
            left["gsis_id"] = pd.NA
            results.append(left)
            continue
        snapshots = snapshots.sort_values("snapshot_utc")
        keep = ["snapshot_utc"] + (["gsis_id"] if "gsis_id" in snapshots.columns else [])
        matched = pd.merge_asof(
            left,
            snapshots[keep],
            left_on="cutoff_utc",
            right_on="snapshot_utc",
            direction="backward",
            allow_exact_matches=True,
        )
        if "gsis_id" not in matched.columns:
            matched["gsis_id"] = pd.NA
        results.append(matched)

    matched_games = pd.concat(results, ignore_index=True, sort=False) if results else matured.copy()
    if "snapshot_utc" not in matched_games.columns:
        matched_games["snapshot_utc"] = pd.NaT
    if "gsis_id" not in matched_games.columns:
        matched_games["gsis_id"] = pd.NA
    matched_games["qb1_at_cutoff"] = matched_games["snapshot_utc"].notna()
    matched_games["qb1_gsis_at_cutoff"] = matched_games["qb1_at_cutoff"] & matched_games["gsis_id"].notna()
    staleness = (
        (matched_games["cutoff_utc"] - matched_games["snapshot_utc"]).dt.total_seconds() / 3600.0
    )
    finite_staleness = staleness[np.isfinite(staleness)]

    summary.update(
        {
            "strict_cutoff_enforceable": bool(recent["snapshot_utc"].notna().mean() >= 0.95),
            "qb1_team_game_coverage_at_cutoff": float(matched_games["qb1_at_cutoff"].mean()) if len(matched_games) else float("nan"),
            "qb1_gsis_coverage_at_cutoff": float(matched_games["qb1_gsis_at_cutoff"].mean()) if len(matched_games) else float("nan"),
            "median_snapshot_staleness_hours": float(finite_staleness.median()) if len(finite_staleness) else None,
            "p90_snapshot_staleness_hours": float(finite_staleness.quantile(0.9)) if len(finite_staleness) else None,
            "by_season": _by_season_rates(
                matched_games,
                season_column="season",
                value_columns=("qb1_at_cutoff", "qb1_gsis_at_cutoff"),
            ),
        }
    )
    return summary


def audit_pregame_context_frames(
    schedules: pd.DataFrame,
    injuries: pd.DataFrame,
    depth_charts: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    cutoff_hours: float = 24.0,
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Audit whether public player-context sources can support strict pregame features."""
    if end_season < start_season:
        raise ValueError("end_season must be >= start_season")
    if cutoff_hours < 0:
        raise ValueError("cutoff_hours must be non-negative")

    as_of_utc = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if as_of_utc.tzinfo is None:
        as_of_utc = as_of_utc.tz_localize("UTC")
    else:
        as_of_utc = as_of_utc.tz_convert("UTC")

    requested = list(range(int(start_season), int(end_season) + 1))
    schedule = schedules.loc[
        pd.to_numeric(schedules["season"], errors="coerce").between(start_season, end_season)
    ].copy()
    team_games = _team_games(schedule, cutoff_hours=cutoff_hours, as_of_utc=as_of_utc)
    matured = team_games.loc[team_games["cutoff_matured"]]

    kickoff_parse_rate = float(team_games["kickoff_utc"].notna().mean()) if len(team_games) else float("nan")
    return {
        "requested_window": {
            "start_season": int(start_season),
            "end_season": int(end_season),
        },
        "as_of_utc": as_of_utc.isoformat(),
        "cutoff_policy": {
            "hours_before_kickoff": float(cutoff_hours),
            "schedule_time_interpretation": "nflverse gameday + gametime interpreted in America/New_York, then converted to UTC",
        },
        "schedule": {
            "games": int(len(schedule)),
            "team_games": int(len(team_games)),
            "matured_team_games": int(len(matured)),
            "kickoff_parse_rate": kickoff_parse_rate,
        },
        "injuries": _injury_audit(injuries, team_games, requested_seasons=requested),
        "depth_charts": {
            "pre_2025_weekly": _pre2025_depth_audit(depth_charts, team_games),
            "post_2025_timestamped": _post2025_depth_audit(depth_charts, team_games),
        },
        "research_policy": {
            "pre_2025_depth_charts": "audit/joinability only; do not call them strict fixed-cutoff features because the records lack timestamps",
            "post_2025_depth_charts": "eligible for fixed-cutoff research only when dt <= game cutoff",
            "injuries": "eligible through 2024 only when date_modified <= game cutoff; upstream has no 2025+ feed",
        },
    }


def audit_pregame_context(
    *,
    start_season: int,
    end_season: int,
    cutoff_hours: float = 24.0,
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    seasons = range(int(start_season), int(end_season) + 1)
    schedules, injuries, depth_charts = load_pregame_context_sources(seasons)
    return audit_pregame_context_frames(
        schedules,
        injuries,
        depth_charts,
        start_season=start_season,
        end_season=end_season,
        cutoff_hours=cutoff_hours,
        as_of=as_of,
    )
