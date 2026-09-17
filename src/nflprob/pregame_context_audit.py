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
    """Load schedules, injuries, and old/new-schema depth charts."""
    try:
        import nflreadpy as nfl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install data dependencies with: pip install 'nflprob[data]'") from exc

    requested = sorted({int(season) for season in seasons})
    if not requested:
        raise ValueError("at least one season is required")

    schedules = _to_pandas(nfl.load_schedules(requested))
    injury_frames: list[pd.DataFrame] = []
    depth_frames: list[pd.DataFrame] = []
    for season in requested:
        if season <= INJURY_LAST_AVAILABLE_SEASON:
            injuries = _load_with_retries(nfl.load_injuries, season)
            if "season" not in injuries.columns:
                injuries["season"] = season
            injury_frames.append(injuries)
        depth = _load_with_retries(nfl.load_depth_charts, season)
        depth["_requested_season"] = season
        depth_frames.append(depth)

    injuries = (
        pd.concat(injury_frames, ignore_index=True, sort=False)
        if injury_frames
        else pd.DataFrame()
    )
    depth_charts = pd.concat(depth_frames, ignore_index=True, sort=False)
    return schedules, injuries, depth_charts


def _parse_kickoff_utc(schedules: pd.DataFrame) -> pd.Series:
    if not {"gameday", "gametime"}.issubset(schedules.columns):
        return pd.Series(pd.NaT, index=schedules.index, dtype="datetime64[ns, UTC]")
    text = schedules["gameday"].astype("string") + " " + schedules["gametime"].astype("string")
    local = pd.to_datetime(text, errors="coerce")
    return local.dt.tz_localize(
        EASTERN_TIMEZONE,
        ambiguous="NaT",
        nonexistent="shift_forward",
    ).dt.tz_convert("UTC")


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
    games["cutoff_matured"] = games["cutoff_utc"].notna() & (
        games["cutoff_utc"] <= as_of_utc
    )
    ids = [column for column in ("game_id", "game_type") if column in games.columns]
    common = ["season", "week", "kickoff_utc", "cutoff_utc", "cutoff_matured", *ids]
    home = games[common + ["home_team"]].rename(columns={"home_team": "team"})
    away = games[common + ["away_team"]].rename(columns={"away_team": "team"})
    home["side"] = "home"
    away["side"] = "away"
    output = pd.concat([home, away], ignore_index=True, sort=False)
    output["team"] = output["team"].astype("string")
    return output


def _by_season_rates(
    frame: pd.DataFrame,
    *,
    value_columns: tuple[str, ...],
) -> list[dict[str, float | int]]:
    if frame.empty or "season" not in frame.columns:
        return []
    seasons = pd.to_numeric(frame["season"], errors="coerce")
    output: list[dict[str, float | int]] = []
    for season in sorted(int(value) for value in seasons.dropna().unique()):
        subset = frame.loc[seasons == season]
        row: dict[str, float | int] = {"season": season, "team_games": len(subset)}
        for column in value_columns:
            if column in subset.columns:
                row[f"{column}_rate"] = float(
                    subset[column].astype("boolean").fillna(False).mean()
                )
        output.append(row)
    return output


def _injury_audit(
    injuries: pd.DataFrame,
    team_games: pd.DataFrame,
    requested_seasons: list[int],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "upstream_last_available_season": INJURY_LAST_AVAILABLE_SEASON,
        "requested_seasons_without_upstream_data": [
            season for season in requested_seasons if season > INJURY_LAST_AVAILABLE_SEASON
        ],
        "rows": len(injuries),
        "columns": sorted(str(column) for column in injuries.columns),
    }
    if injuries.empty:
        summary["strict_cutoff_ready"] = False
        return summary
    required = {"season", "week", "team"}
    if not required.issubset(injuries.columns):
        summary.update(
            strict_cutoff_ready=False,
            error="injury data lacks season/week/team join keys",
        )
        return summary

    frame = injuries.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
    frame["team"] = frame["team"].astype("string")
    modified = (
        pd.to_datetime(frame["date_modified"], errors="coerce", utc=True)
        if "date_modified" in frame.columns
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    )
    frame["modified_utc"] = modified

    matured = team_games.loc[
        team_games["cutoff_matured"]
        & pd.to_numeric(team_games["season"], errors="coerce").le(2024)
    ].copy()
    keys = ["season", "week", "team"]
    records = frame.merge(matured[keys + ["cutoff_utc"]], on=keys, how="inner")
    records["eligible_at_cutoff"] = records["modified_utc"].notna() & (
        records["modified_utc"] <= records["cutoff_utc"]
    )
    any_report = records[keys].drop_duplicates().assign(any_report=True)
    eligible = (
        records.loc[records["eligible_at_cutoff"], keys]
        .drop_duplicates()
        .assign(eligible_report=True)
    )
    joined = matured.merge(any_report, on=keys, how="left").merge(
        eligible,
        on=keys,
        how="left",
    )
    joined["any_report"] = joined["any_report"].astype("boolean").fillna(False).astype(bool)
    joined["eligible_report"] = (
        joined["eligible_report"].astype("boolean").fillna(False).astype(bool)
    )

    player_key = "gsis_id" if "gsis_id" in frame.columns else None
    revision_groups = 0
    if player_key:
        revision_groups = int(
            (frame.groupby(keys + [player_key], dropna=False).size() > 1).sum()
        )
    position = frame.get("position", pd.Series(index=frame.index, dtype="string"))
    qb_rows = position.astype("string").str.upper().eq("QB")

    summary.update(
        {
            "date_modified_present": "date_modified" in frame.columns,
            "date_modified_parse_rate": float(modified.notna().mean()),
            "seasons_present": sorted(int(value) for value in frame["season"].dropna().unique()),
            "teams_present": int(frame["team"].nunique(dropna=True)),
            "players_present": int(frame[player_key].nunique(dropna=True)) if player_key else None,
            "qb_rows": int(qb_rows.sum()),
            "revision_player_week_groups": revision_groups,
            "matured_team_games": len(matured),
            "team_game_any_report_rate": (
                float(joined["any_report"].mean()) if len(joined) else float("nan")
            ),
            "team_game_report_available_at_cutoff_rate": (
                float(joined["eligible_report"].mean()) if len(joined) else float("nan")
            ),
            "by_season": _by_season_rates(
                joined,
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


def _pre2025_depth_audit(
    depth: pd.DataFrame,
    team_games: pd.DataFrame,
) -> dict[str, object]:
    historical = depth.loc[
        pd.to_numeric(depth["_requested_season"], errors="coerce").le(2024)
    ].copy()
    matured = team_games.loc[
        team_games["cutoff_matured"]
        & pd.to_numeric(team_games["season"], errors="coerce").le(2024)
    ].copy()
    summary: dict[str, object] = {
        "rows": len(historical),
        "matured_team_games": len(matured),
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
        historical.get("season", historical["_requested_season"]),
        errors="coerce",
    ).astype("Int64")
    historical["week_key"] = pd.to_numeric(historical["week"], errors="coerce").astype("Int64")
    historical["team_key"] = historical[team_column].astype("string")
    position = historical.get(
        "position",
        pd.Series(index=historical.index, dtype="string"),
    )
    qbs = historical.loc[position.astype("string").str.upper().eq("QB")].copy()
    if "depth_team" in qbs.columns:
        qbs = qbs.loc[pd.to_numeric(qbs["depth_team"], errors="coerce").eq(1)]
    available = (
        qbs[["season_key", "week_key", "team_key"]]
        .drop_duplicates()
        .assign(qb1_weekly_available=True)
    )
    joined = matured.assign(
        season_key=pd.to_numeric(matured["season"], errors="coerce").astype("Int64"),
        week_key=pd.to_numeric(matured["week"], errors="coerce").astype("Int64"),
        team_key=matured["team"].astype("string"),
    ).merge(available, on=["season_key", "week_key", "team_key"], how="left")
    joined["qb1_weekly_available"] = (
        joined["qb1_weekly_available"].astype("boolean").fillna(False).astype(bool)
    )
    summary.update(
        {
            "qb1_weekly_team_game_coverage": (
                float(joined["qb1_weekly_available"].mean()) if len(joined) else float("nan")
            ),
            "by_season": _by_season_rates(
                joined,
                value_columns=("qb1_weekly_available",),
            ),
            "interpretation": (
                "joinable by season/week/team, but no record timestamp proves the snapshot "
                "was available by a fixed pregame cutoff"
            ),
        }
    )
    return summary


def _post2025_depth_audit(
    depth: pd.DataFrame,
    team_games: pd.DataFrame,
) -> dict[str, object]:
    recent = depth.loc[
        pd.to_numeric(depth["_requested_season"], errors="coerce").ge(2025)
    ].copy()
    matured = team_games.loc[
        team_games["cutoff_matured"]
        & pd.to_numeric(team_games["season"], errors="coerce").ge(2025)
    ].copy()
    summary: dict[str, object] = {
        "rows": len(recent),
        "matured_team_games": len(matured),
        "timestamped": "dt" in recent.columns,
    }
    if recent.empty or "dt" not in recent.columns:
        summary["strict_cutoff_enforceable"] = False
        return summary

    recent["snapshot_utc"] = pd.to_datetime(recent["dt"], errors="coerce", utc=True)
    parse_rate = float(recent["snapshot_utc"].notna().mean())
    summary["timestamp_parse_rate"] = parse_rate
    if not {"team", "pos_abb"}.issubset(recent.columns):
        summary.update(
            strict_cutoff_enforceable=False,
            error="2025+ depth chart data lacks team/pos_abb fields",
        )
        return summary
    if matured.empty:
        summary.update(
            {
                "strict_cutoff_enforceable": bool(parse_rate >= 0.95),
                "qb1_team_game_coverage_at_cutoff": float("nan"),
                "qb1_gsis_coverage_at_cutoff": float("nan"),
                "median_snapshot_staleness_hours": None,
                "p90_snapshot_staleness_hours": None,
                "by_season": [],
            }
        )
        return summary

    recent["season_key"] = pd.to_numeric(
        recent["_requested_season"],
        errors="coerce",
    ).astype("Int64")
    recent["team_key"] = recent["team"].astype("string")
    qbs = recent.loc[recent["pos_abb"].astype("string").str.upper().eq("QB")].copy()
    if "pos_rank" in qbs.columns:
        qbs["rank"] = pd.to_numeric(qbs["pos_rank"], errors="coerce")
        minimum = qbs.groupby(
            ["season_key", "team_key", "snapshot_utc"]
        )["rank"].transform("min")
        qbs = qbs.loc[qbs["rank"].eq(minimum)]
    qbs = qbs.sort_values(["season_key", "team_key", "snapshot_utc"], kind="stable")
    qbs = qbs.drop_duplicates(["season_key", "team_key", "snapshot_utc"], keep="first")

    matched_frames: list[pd.DataFrame] = []
    for (season, team), game_group in matured.groupby(["season", "team"], dropna=False):
        snapshots = qbs.loc[
            qbs["season_key"].eq(int(season)) & qbs["team_key"].eq(str(team))
        ].copy()
        left = game_group.sort_values("cutoff_utc").copy()
        if snapshots.empty:
            left["snapshot_utc"] = pd.Series(
                pd.NaT,
                index=left.index,
                dtype="datetime64[ns, UTC]",
            )
            left["gsis_id"] = pd.NA
            matched_frames.append(left)
            continue
        keep = ["snapshot_utc"] + (["gsis_id"] if "gsis_id" in snapshots.columns else [])
        matched = pd.merge_asof(
            left,
            snapshots[keep].sort_values("snapshot_utc"),
            left_on="cutoff_utc",
            right_on="snapshot_utc",
            direction="backward",
            allow_exact_matches=True,
        )
        if "gsis_id" not in matched.columns:
            matched["gsis_id"] = pd.NA
        matched_frames.append(matched)

    matched_games = pd.concat(matched_frames, ignore_index=True, sort=False)
    matched_games["snapshot_utc"] = pd.to_datetime(
        matched_games["snapshot_utc"],
        errors="coerce",
        utc=True,
    )
    matched_games["qb1_at_cutoff"] = matched_games["snapshot_utc"].notna()
    matched_games["qb1_gsis_at_cutoff"] = (
        matched_games["qb1_at_cutoff"] & matched_games["gsis_id"].notna()
    )
    staleness = (
        (matched_games["cutoff_utc"] - matched_games["snapshot_utc"]).dt.total_seconds()
        / 3600.0
    )
    finite = staleness[np.isfinite(staleness)]
    summary.update(
        {
            "strict_cutoff_enforceable": bool(parse_rate >= 0.95),
            "qb1_team_game_coverage_at_cutoff": float(
                matched_games["qb1_at_cutoff"].mean()
            ),
            "qb1_gsis_coverage_at_cutoff": float(
                matched_games["qb1_gsis_at_cutoff"].mean()
            ),
            "median_snapshot_staleness_hours": (
                float(finite.median()) if len(finite) else None
            ),
            "p90_snapshot_staleness_hours": (
                float(finite.quantile(0.9)) if len(finite) else None
            ),
            "by_season": _by_season_rates(
                matched_games,
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
    as_of_utc = (
        as_of_utc.tz_localize("UTC")
        if as_of_utc.tzinfo is None
        else as_of_utc.tz_convert("UTC")
    )
    requested = list(range(int(start_season), int(end_season) + 1))
    schedule = schedules.loc[
        pd.to_numeric(schedules["season"], errors="coerce").between(
            start_season,
            end_season,
        )
    ].copy()
    team_games = _team_games(schedule, cutoff_hours=cutoff_hours, as_of_utc=as_of_utc)
    matured = team_games.loc[team_games["cutoff_matured"]]
    kickoff_rate = (
        float(team_games["kickoff_utc"].notna().mean())
        if len(team_games)
        else float("nan")
    )
    return {
        "requested_window": {
            "start_season": int(start_season),
            "end_season": int(end_season),
        },
        "as_of_utc": as_of_utc.isoformat(),
        "cutoff_policy": {
            "hours_before_kickoff": float(cutoff_hours),
            "schedule_time_interpretation": (
                "nflverse gameday + gametime interpreted in America/New_York, then UTC"
            ),
        },
        "schedule": {
            "games": len(schedule),
            "team_games": len(team_games),
            "matured_team_games": len(matured),
            "kickoff_parse_rate": kickoff_rate,
        },
        "injuries": _injury_audit(injuries, team_games, requested),
        "depth_charts": {
            "pre_2025_weekly": _pre2025_depth_audit(depth_charts, team_games),
            "post_2025_timestamped": _post2025_depth_audit(depth_charts, team_games),
        },
        "research_policy": {
            "pre_2025_depth_charts": (
                "joinability only; no timestamp proves a fixed-cutoff snapshot"
            ),
            "post_2025_depth_charts": (
                "eligible only when depth-chart dt is at or before game cutoff"
            ),
            "injuries": (
                "eligible through 2024 only when date_modified is at or before cutoff; "
                "upstream has no 2025+ feed"
            ),
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
