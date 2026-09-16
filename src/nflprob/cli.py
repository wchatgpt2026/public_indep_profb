from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import build_nflverse_dataset
from .evaluation import evaluate_holdout
from .model import NFLPredictor


def _read_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError("data file must be .parquet or .csv")


def _write_frame(frame: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        frame.to_csv(p, index=False)
    else:
        raise ValueError("output must be .parquet or .csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nflprob", description="Independent NFL probabilistic model"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("build-data", help="download nflverse data and build pregame features")
    p_data.add_argument("--start-season", type=int, required=True)
    p_data.add_argument("--end-season", type=int, required=True)
    p_data.add_argument("--output", required=True)
    p_data.add_argument("--half-life", type=float, default=6.0)

    p_train = sub.add_parser("train", help="fit a model artifact")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)
    p_train.add_argument("--score-max", type=int, default=80)

    p_price = sub.add_parser("price", help="price one game from a prepared feature dataset")
    p_price.add_argument("--model", required=True)
    p_price.add_argument("--data", required=True)
    p_price.add_argument("--game-id", required=True)
    p_price.add_argument("--spread", type=float)
    p_price.add_argument("--total", type=float)
    p_price.add_argument("--distribution-csv")

    p_eval = sub.add_parser("evaluate", help="evaluate a fitted artifact on a holdout file")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--data", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-data":
        seasons = range(args.start_season, args.end_season + 1)
        frame = build_nflverse_dataset(seasons, half_life_games=args.half_life)
        _write_frame(frame, args.output)
        print(json.dumps({"rows": len(frame), "output": args.output}, indent=2))
        return 0
    if args.command == "train":
        frame = _read_frame(args.data)
        model = NFLPredictor(score_max=args.score_max).fit(frame)
        model.save(args.model)
        print(json.dumps({"model": args.model, "training_metrics": model.training_metrics_}, indent=2))
        return 0
    if args.command == "price":
        model = NFLPredictor.load(args.model)
        frame = _read_frame(args.data)
        matches = frame.loc[frame["game_id"].astype(str) == str(args.game_id)]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one row for game_id={args.game_id!r}; found {len(matches)}"
            )
        row = matches.iloc[0]
        print(json.dumps(model.price_game(row, home_spread=args.spread, total=args.total), indent=2))
        if args.distribution_csv:
            model.predict_distribution(row).to_frame().to_csv(args.distribution_csv, index=False)
        return 0
    if args.command == "evaluate":
        model = NFLPredictor.load(args.model)
        frame = _read_frame(args.data)
        print(json.dumps(evaluate_holdout(model, frame), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
