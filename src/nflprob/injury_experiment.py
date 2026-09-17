from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .development import (
    DEVELOPMENT_MAX_SEASON,
    SELECTION_METRICS,
    _comparison_improvements,
    _model_summary,
    _normalized_selection_loss,
    _run_feature_backtest,
    _strip_market_columns,
)
from .model import default_feature_columns
from .pregame_context_audit import (
    INJURY_LAST_AVAILABLE_SEASON,
    _load_with_retries,
    _team_games,
    _to_pandas,
)


INJURY_GROUP_METRICS = {
    "practice_load": (
        "t24_inj_player_count",
        "t24_inj_practice_dnp_count",
        "t24_inj_practice_limited_count",
        "t24_inj_practice_burden",
    ),
    "game_status": (
        "t24_inj_status_out_count",
        "t24_inj_status_doubtful_count",
        "t24_inj_status_questionable_count",
        "t24_inj_status_burden",
    ),
    "position_concentration": (
        "t24_inj_qb_practice_burden",
        "t24_inj_qb_status_burden",
        "t24_inj_ol_practice_burden",
        "t24_inj_ol_status_burden",
        "t24_inj_skill_practice_burden",
        "t24_inj_skill_status_burden",
        "t24_inj_defense_practice_burden",
        "t24_inj_defense_status_burden",
    ),
}
INJURY_TEAM_METRICS = tuple(
    metric for metrics in INJURY_GROUP_METRICS.values() for metric in metrics
)
OL_POSITIONS = frozenset({"C", "G", "OG", "OT", "T", "OL"})
SKILL_POSITIONS = frozenset({"RB", "FB", "WR", "TE"})
DEFENSE_POSITIONS = frozenset(
    {"DE", "DT", "DL", "NT", "LB", "ILB", "OLB", "CB", "DB", "S", "FS", "SS"}
)


def load_injury_context_sources(
    seasons: list[int] | range,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load schedules plus the timestamped nflverse injury feed through 2024."""
    try:
        import nflreadpy as nfl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install data dependencies with: pip install 'nflprob[data]'") from exc

    requested = sorted({int(season) for season in seasons})
    if not requested:
        raise ValueError("at least one season is required")
    if max(requested) > INJURY_LAST_AVAILABLE_SEASON:
        raise ValueError("injury development data is available only through the 2024 season")

    schedules = _to_pandas(nfl.load_schedules(requested))
    frames: list[pd.DataFrame] = []
    for season in requested:
        frame = _load_with_retries(nfl.load_injuries, season)
        if "season" not in frame.columns:
            frame["season"] = season
        frames.append(frame)
    injuries = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return schedules, injuries


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower()


def _player_key(frame: pd.DataFrame) -> pd.Series:
    key = (
        frame["gsis_id"].astype("string")
        if "gsis_id" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    if "full_name" in frame.columns:
        key = key.fillna(frame["full_name"].astype("string"))
    missing = key.isna() | key.eq("")
    if missing.any():
        replacement = pd.Series(
            [f"__row_{index}" for index in frame.index],
            index=frame.index,
            dtype="string",
        )
        key = key.where(~missing, replacement)
    return key


def _position_group_flags(position: pd.Series) -> dict[str, pd.Series]:
    pos = position.astype("string").fillna("").str.upper()
    return {
        "qb": pos.eq("QB"),
        "ol": pos.isin(OL_POSITIONS),
        "skill": pos.isin(SKILL_POSITIONS),
        "defense": pos.isin(DEFENSE_POSITIONS),
    }


def build_t24_injury_features(
    schedules: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    cutoff_hours: float = 24.0,
) -> pd.DataFrame:
    """Build one leakage-safe injury/practice feature row per scheduled game.

    For each team-game, only injury records whose ``date_modified`` is at or before the
    fixed pregame cutoff are eligible. If a player has multiple eligible revisions, the
    latest eligible revision is used.
    """
    if cutoff_hours < 0:
        raise ValueError("cutoff_hours must be non-negative")
    if "game_id" not in schedules.columns:
        raise ValueError("schedule data must contain game_id")
    required = {"season", "week", "team", "date_modified", "practice_status"}
    missing = required - set(injuries.columns)
    if missing:
        raise ValueError(f"injury data missing required columns: {sorted(missing)}")

    far_future = pd.Timestamp("2100-01-01T00:00:00Z")
    team_games = _team_games(
        schedules,
        cutoff_hours=cutoff_hours,
        as_of_utc=far_future,
    )
    team_games = team_games.loc[
        pd.to_numeric(team_games["season"], errors="coerce").le(INJURY_LAST_AVAILABLE_SEASON)
    ].copy()
    keys = ["season", "week", "team"]

    frame = injuries.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
    frame["team"] = frame["team"].astype("string")
    frame["modified_utc"] = pd.to_datetime(frame["date_modified"], errors="coerce", utc=True)
    frame["player_key"] = _player_key(frame)

    records = frame.merge(
        team_games[[*keys, "game_id", "side", "cutoff_utc"]],
        on=keys,
        how="inner",
    )
    records = records.loc[
        records["modified_utc"].notna() & (records["modified_utc"] <= records["cutoff_utc"])
    ].copy()
    records = records.sort_values(
        ["game_id", "side", "player_key", "modified_utc"],
        kind="stable",
    ).drop_duplicates(["game_id", "side", "player_key"], keep="last")

    practice = _normalize_text(records["practice_status"])
    report = _normalize_text(
        records.get("report_status", pd.Series(index=records.index, dtype="string"))
    )
    records["practice_dnp"] = practice.str.contains("did not participate") | practice.eq("dnp")
    records["practice_limited"] = practice.str.contains("limited")
    records["practice_burden"] = (
        2.0 * records["practice_dnp"].astype(float)
        + records["practice_limited"].astype(float)
    )
    records["status_out"] = report.eq("out")
    records["status_doubtful"] = report.eq("doubtful")
    records["status_questionable"] = report.eq("questionable")
    records["status_burden"] = (
        3.0 * records["status_out"].astype(float)
        + 2.0 * records["status_doubtful"].astype(float)
        + records["status_questionable"].astype(float)
    )

    flags = _position_group_flags(
        records.get("position", pd.Series(index=records.index, dtype="string"))
    )
    for name, mask in flags.items():
        records[f"{name}_practice_burden"] = records["practice_burden"] * mask.astype(float)
        records[f"{name}_status_burden"] = records["status_burden"] * mask.astype(float)

    if records.empty:
        aggregates = pd.DataFrame(columns=["game_id", "side", *INJURY_TEAM_METRICS])
    else:
        grouped = records.groupby(["game_id", "side"], dropna=False)
        aggregates = grouped.agg(
            t24_inj_player_count=("player_key", "size"),
            t24_inj_practice_dnp_count=("practice_dnp", "sum"),
            t24_inj_practice_limited_count=("practice_limited", "sum"),
            t24_inj_practice_burden=("practice_burden", "sum"),
            t24_inj_status_out_count=("status_out", "sum"),
            t24_inj_status_doubtful_count=("status_doubtful", "sum"),
            t24_inj_status_questionable_count=("status_questionable", "sum"),
            t24_inj_status_burden=("status_burden", "sum"),
            t24_inj_qb_practice_burden=("qb_practice_burden", "sum"),
            t24_inj_qb_status_burden=("qb_status_burden", "sum"),
            t24_inj_ol_practice_burden=("ol_practice_burden", "sum"),
            t24_inj_ol_status_burden=("ol_status_burden", "sum"),
            t24_inj_skill_practice_burden=("skill_practice_burden", "sum"),
            t24_inj_skill_status_burden=("skill_status_burden", "sum"),
            t24_inj_defense_practice_burden=("defense_practice_burden", "sum"),
            t24_inj_defense_status_burden=("defense_status_burden", "sum"),
        ).reset_index()

    base = team_games[["game_id", "side"]].drop_duplicates().merge(
        aggregates,
        on=["game_id", "side"],
        how="left",
    )
    for metric in INJURY_TEAM_METRICS:
        base[metric] = pd.to_numeric(base[metric], errors="coerce").fillna(0.0).astype(float)

    home = base.loc[base["side"].eq("home"), ["game_id", *INJURY_TEAM_METRICS]].rename(
        columns={metric: f"home_{metric}" for metric in INJURY_TEAM_METRICS}
    )
    away = base.loc[base["side"].eq("away"), ["game_id", *INJURY_TEAM_METRICS]].rename(
        columns={metric: f"away_{metric}" for metric in INJURY_TEAM_METRICS}
    )
    wide = home.merge(away, on="game_id", how="outer", validate="one_to_one")
    for metric in INJURY_TEAM_METRICS:
        home_column = f"home_{metric}"
        away_column = f"away_{metric}"
        wide[f"diff_{metric}"] = wide[home_column] - wide[away_column]
        wide[f"sum_{metric}"] = wide[home_column] + wide[away_column]
    return wide


def injury_feature_group(column: str) -> str | None:
    for group, metrics in INJURY_GROUP_METRICS.items():
        if any(column.endswith(metric) for metric in metrics):
            return group
    return None


def injury_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if injury_feature_group(column) is not None]


def _columns_by_group(columns: list[str]) -> dict[str, list[str]]:
    groups = {group: [] for group in INJURY_GROUP_METRICS}
    for column in columns:
        group = injury_feature_group(column)
        if group is not None:
            groups[group].append(column)
    return groups


def development_injury_group_ablation(
    games: pd.DataFrame,
    schedules: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    selection_start_season: int = 2019,
    selection_end_season: int = 2020,
    validation_season: int = 2021,
    cutoff_hours: float = 24.0,
    min_train_games: int = 500,
    score_max: int = 80,
    random_state: int = 7,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Select T-24 injury feature groups on 2019-2020 and validate once on 2021."""
    if validation_season > DEVELOPMENT_MAX_SEASON:
        raise ValueError("dev-injury-ablate validation is locked to seasons through 2021")
    if selection_end_season >= validation_season:
        raise ValueError("selection_end_season must be earlier than validation_season")
    if selection_start_season > selection_end_season:
        raise ValueError("selection_start_season must be <= selection_end_season")

    base_games = _strip_market_columns(games)
    baseline_columns = default_feature_columns(base_games)
    context = build_t24_injury_features(schedules, injuries, cutoff_hours=cutoff_hours)
    added_columns = injury_feature_columns(context)
    if not added_columns:
        raise ValueError("no T-24 injury features could be constructed")

    augmented = base_games.merge(context, on="game_id", how="left", indicator="_injury_join")
    seasons = pd.to_numeric(augmented["season"], errors="coerce")
    research_rows = seasons.le(validation_season)
    match_rate = float(augmented.loc[research_rows, "_injury_join"].eq("both").mean())
    if not np.isfinite(match_rate) or match_rate < 0.98:
        raise ValueError(f"injury context game_id match rate is too low: {match_rate:.3f}")
    augmented = augmented.drop(columns="_injury_join")
    augmented[added_columns] = augmented[added_columns].fillna(0.0).astype(float)

    group_columns = _columns_by_group(added_columns)
    missing_groups = [name for name, columns in group_columns.items() if not columns]
    if missing_groups:
        raise ValueError("injury feature groups are missing: " + ", ".join(missing_groups))

    baseline_selection_report, _ = _run_feature_backtest(
        augmented,
        feature_columns=baseline_columns,
        start_season=selection_start_season,
        end_season=selection_end_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state,
    )
    baseline_selection = _model_summary(baseline_selection_report)

    candidate_results: list[dict[str, object]] = []
    best_groups: tuple[str, ...] = ()
    best_loss = 1.0
    best_added_count = 0
    group_names = tuple(INJURY_GROUP_METRICS)
    for size in range(1, len(group_names) + 1):
        for selected_groups in combinations(group_names, size):
            columns = [
                column
                for group in selected_groups
                for column in group_columns[group]
            ]
            candidate_report, _ = _run_feature_backtest(
                augmented,
                feature_columns=baseline_columns + columns,
                start_season=selection_start_season,
                end_season=selection_end_season,
                min_train_games=min_train_games,
                score_max=score_max,
                random_state=random_state,
            )
            summary = _model_summary(candidate_report)
            loss = _normalized_selection_loss(summary, baseline_selection)
            candidate_results.append(
                {
                    "groups": list(selected_groups),
                    "added_features": len(columns),
                    "selection_loss_ratio": loss,
                    "balanced_improvement_pct": (
                        (1.0 - loss) * 100.0 if np.isfinite(loss) else float("nan")
                    ),
                    "summary": summary,
                    "vs_baseline_improvement_pct": _comparison_improvements(
                        summary,
                        baseline_selection,
                    ),
                }
            )
            if loss < best_loss - 1e-12 or (
                abs(loss - best_loss) <= 1e-12 and len(columns) < best_added_count
            ):
                best_groups = selected_groups
                best_loss = loss
                best_added_count = len(columns)

    winner_columns = [
        column for group in best_groups for column in group_columns[group]
    ]
    baseline_validation_report, _ = _run_feature_backtest(
        augmented,
        feature_columns=baseline_columns,
        start_season=validation_season,
        end_season=validation_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state + 100,
    )
    winner_validation_report, winner_predictions = _run_feature_backtest(
        augmented,
        feature_columns=baseline_columns + winner_columns,
        start_season=validation_season,
        end_season=validation_season,
        min_train_games=min_train_games,
        score_max=score_max,
        random_state=random_state + 100,
    )
    baseline_validation = _model_summary(baseline_validation_report)
    winner_validation = _model_summary(winner_validation_report)

    candidate_results.sort(
        key=lambda row: (float(row["selection_loss_ratio"]), int(row["added_features"]))
    )
    report: dict[str, object] = {
        "selection_window": {
            "start_season": int(selection_start_season),
            "end_season": int(selection_end_season),
        },
        "validation_season": int(validation_season),
        "reserved_confirmation_starts": DEVELOPMENT_MAX_SEASON + 1,
        "baseline_policy": "78-feature deployable pregame-only defaults",
        "context_policy": {
            "cutoff_hours_before_kickoff": float(cutoff_hours),
            "record_rule": "latest player injury record with date_modified <= game cutoff",
            "upstream_last_available_season": INJURY_LAST_AVAILABLE_SEASON,
        },
        "context_game_id_match_rate": match_rate,
        "selection_metrics": list(SELECTION_METRICS),
        "feature_groups": {
            name: {"features": columns, "count": len(columns)}
            for name, columns in group_columns.items()
        },
        "baseline_selection_summary": baseline_selection,
        "candidates": candidate_results,
        "winner": {
            "groups": list(best_groups),
            "added_features": len(winner_columns),
            "selection_loss_ratio": best_loss,
            "balanced_improvement_pct": (1.0 - best_loss) * 100.0,
        },
        "validation": {
            "baseline_summary": baseline_validation,
            "winner_summary": winner_validation,
            "winner_vs_baseline_improvement_pct": _comparison_improvements(
                winner_validation,
                baseline_validation,
            ),
        },
    }
    return report, winner_predictions


def development_injury_group_ablation_live(
    games: pd.DataFrame,
    **kwargs: object,
) -> tuple[dict[str, object], pd.DataFrame]:
    validation_season = int(kwargs.get("validation_season", 2021))
    seasons = pd.to_numeric(games["season"], errors="coerce").dropna().astype(int)
    if seasons.empty:
        raise ValueError("games data has no valid seasons")
    first_season = int(seasons.min())
    source_seasons = list(range(first_season, validation_season + 1))
    schedules, injuries = load_injury_context_sources(source_seasons)
    return development_injury_group_ablation(
        games,
        schedules,
        injuries,
        **kwargs,
    )
