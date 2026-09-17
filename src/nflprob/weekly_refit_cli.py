from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .weekly_refit_experiment import development_weekly_refit_compare


def _read_frame(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError("data file must be .parquet or .csv")


def _write_frame(frame: pd.DataFrame, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(output, index=False)
    elif output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
    else:
        raise ValueError("output must be .parquet or .csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nflprob-dev-weekly-refit",
        description="Development-only comparison of season-frozen vs weekly expanding refits",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--selection-start-season", type=int, default=2019)
    parser.add_argument("--selection-end-season", type=int, default=2020)
    parser.add_argument("--validation-season", type=int, default=2021)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--score-max", type=int, default=80)
    parser.add_argument("--predictions-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = _read_frame(args.data)
    report, predictions = development_weekly_refit_compare(
        frame,
        selection_start_season=args.selection_start_season,
        selection_end_season=args.selection_end_season,
        validation_season=args.validation_season,
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
