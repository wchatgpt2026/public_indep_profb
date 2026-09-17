from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .injury_margin_confirmation import confirmation_injury_margin_compare_live


def _read_frame(path: str) -> pd.DataFrame:
    target = Path(path)
    if target.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(target)
    if target.suffix.lower() == ".csv":
        return pd.read_csv(target)
    raise ValueError("data file must be .parquet or .csv")


def _write_frame(frame: pd.DataFrame, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(target, index=False)
    elif target.suffix.lower() == ".csv":
        frame.to_csv(target, index=False)
    else:
        raise ValueError("output must be .parquet or .csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nflprob-confirm-injury-margin",
        description="One-shot 2022-2024 confirmation of the fixed margin-only injury candidate",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--cutoff-hours", type=float, default=24.0)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--score-max", type=int, default=80)
    parser.add_argument("--predictions-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = _read_frame(args.data)
    report, predictions = confirmation_injury_margin_compare_live(
        frame,
        start_season=args.start_season,
        end_season=args.end_season,
        cutoff_hours=args.cutoff_hours,
        min_train_games=args.min_train_games,
        score_max=args.score_max,
    )
    if args.predictions_output:
        _write_frame(predictions, args.predictions_output)
        report["predictions_output"] = args.predictions_output
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
