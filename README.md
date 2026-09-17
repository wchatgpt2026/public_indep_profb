# public_indep_profb

Independent, research-grade NFL forecasting package that produces a **full joint final-score distribution** and can therefore price moneylines, spreads, totals, and exact scores from one internally coherent probability surface.

The design is intentionally market-independent: sportsbook lines and odds are blocked from the predictive feature matrix. Public nflverse play-by-play and schedule data provide the default inputs.

## What is implemented

- Leakage-safe prior-game team features from EPA/play, success rate, dropback/rush EPA, explosives, turnovers, sacks, CPOE, points, and rest.
- Neutral-site status and an internally maintained Elo rating.
- An experimental, non-default pace/scoring block with prior-game snaps, drives, plays/drive, within-drive seconds/snap, no-huddle rate, early-down pass rate, red-zone EPA/success and scoring-drive rate, plus opponent-side counterparts.
- Exponentially weighted team form with current-game data shifted out of every pregame row.
- Separate **margin** and **total** ensembles: ridge + histogram gradient boosting + Extra Trees, with chronological validation used to learn non-negative blend weights.
- Expanding-window out-of-fold forecasts for distribution calibration.
- A discrete **joint score distribution** based on ex-ante analog games plus recency weighting and maximum-entropy tilting to the point model's target home/away means.
- Fair moneyline, spread and total pricing with push handling and fair decimal/American odds.
- Season-by-season rolling-origin backtesting with calibration, naive historical baselines, and evaluation-only nflverse market benchmarks.
- Guarded development workflows that use the same deployable feature policy as production and keep 2022+ outside feature selection.
- A data-audit command that reports QB/weather/market coverage without fitting a model.
- CLI, serialization, holdout evaluation, unit tests and GitHub Actions CI.

## Operational pregame policy

The prepared parquet may contain historical nflverse fields that are useful for analysis but are **not valid as-of-prediction inputs** for a deployable backtest.

The default model therefore excludes these 14 fields:

- historical starter-derived QB context: `home_qb_epa`, `away_qb_epa`, `home_qb_cpoe`, `away_qb_cpoe`, `home_qb_experience`, `away_qb_experience`, `home_qb_known`, `away_qb_known`, `qb_epa_diff`, `qb_cpoe_diff`, `qb_experience_diff`
- realized game-condition context: `temperature`, `wind_speed`, `indoors`

The historical nflverse schedule QB identifiers are reconstructed from the quarterback who actually played in the game, and the weather/roof fields describe realized game conditions. Those columns can remain in the dataset for audit and future research, but `NFLPredictor` excludes them automatically and rejects them if they are manually passed as predictive features.

On the current 2016-2026 dataset this changes the accepted default feature count from 92 to **78**. A future QB layer should use a timestamped pregame starter/injury source, and weather should use a forecast captured as of prediction time.

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for assumptions and limitations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[data,dev]'
```

## End-to-end run

Build a dataset:

```bash
nflprob build-data \
  --start-season 2016 \
  --end-season 2026 \
  --output data/games.parquet
```

The parquet can retain experimental and audit-only columns without changing normal model behavior.

Audit the actual input coverage:

```bash
nflprob audit-data --data data/games.parquet
```

Train the deployable/default artifact:

```bash
nflprob train \
  --data data/games.parquet \
  --model artifacts/nfl_model.joblib
```

Run a rolling-origin benchmark:

```bash
nflprob backtest \
  --data data/games.parquet \
  --start-season 2022 \
  --end-season 2026 \
  --predictions-output artifacts/walk_forward_predictions.csv
```

Each test season is predicted by a fresh model trained only on completed games from earlier seasons. The report includes aggregate and season-level margin/total MAE and RMSE, home-moneyline Brier score, binary log loss, expected calibration error, exact-score negative log likelihood, and comparison against an expanding-history constant baseline. When nflverse supplies market fields, the same report also compares the independent model with spread, total and no-vig moneyline implied probability on the exact same games. Market fields are retained solely after prediction for evaluation and remain blocked from training.

### Research history and current protocol

The earlier 2019-2021 pace/scoring experiments used the former 92-feature baseline that included historical starter-derived QB context and realized weather. Those results are retained only as research history.

The current `dev-compare`, `dev-ablate`, and `dev-target-split` commands now derive their baseline from the same deployable pregame-only selector used by `NFLPredictor`. Postgame QB/weather fields therefore remain blocked in both production and development research.

For nested pace/scoring selection, use 2019-2020 for selection and 2021 for internal validation:

```bash
nflprob dev-ablate \
  --data data/games.parquet \
  --selection-start-season 2019 \
  --selection-end-season 2020 \
  --validation-season 2021 \
  --predictions-output artifacts/dev_ablation_winner_2021.csv
```

All seven non-empty combinations of the pace/volume, play-calling, and scoring-efficiency blocks are compared against the operational baseline on the selection window. Only the selected winner is evaluated on 2021. Seasons 2022+ remain outside this selection loop and should be used only for later confirmation.

Price a scheduled game already present in the prepared dataset:

```bash
nflprob price \
  --model artifacts/nfl_model.joblib \
  --data data/games.parquet \
  --game-id 2026_02_NYJ_BUF \
  --spread -6.5 \
  --total 44.5 \
  --distribution-csv artifacts/2026_02_NYJ_BUF_scores.csv
```

The JSON response includes projected home/away score, margin, total, two-way moneyline prices, spread prices and total prices. The optional CSV contains every score pair from 0-0 through 80-80 with its probability.

Evaluate on a separate chronologically later holdout file:

```bash
nflprob evaluate --model artifacts/nfl_model.joblib --data data/holdout.parquet
```

## Python API

```python
import pandas as pd
from nflprob.model import NFLPredictor

model = NFLPredictor.load("artifacts/nfl_model.joblib")
games = pd.read_parquet("data/games.parquet")
row = games.loc[games.game_id == "2026_02_NYJ_BUF"].iloc[0]

dist = model.predict_distribution(row)
print(dist.moneyline("home"))
print(dist.spread(-6.5, "home"))
print(dist.total(44.5, "over"))
print(dist.exact_score(27, 20))
```

## Modeling notes

A full score model needs more than a normal approximation to margin. NFL final scores have visible scoring-number structure, home/away residual dependence and heavier tails than independent Poisson models usually imply. This project therefore uses the ML ensemble for the two economically important first moments (margin and total), then calibrates a football-shaped joint discrete distribution from out-of-fold analog games and tilts that distribution to the current means.

That keeps prices internally consistent: moneyline, spread and total are integrations of the **same** joint score matrix rather than separate classifiers that can contradict one another.

## Highest-value next extensions

1. Timestamped starter-QB, injury and practice-participation data with an explicit late-scratch override path.
2. Stadium-specific forecast weather captured as-of prediction time.
3. Offensive-line continuity and skill-position availability features.
4. Nested walk-forward hyperparameter and score-distribution tuning inside development-era data only.
5. Distribution diagnostics such as CRPS, interval coverage and sharpness.
6. Automated weekly data refresh/retrain/publish workflow after the research protocol is accepted.

## Data attribution

Default data access is via the nflverse ecosystem. Review the upstream nflverse licenses and attribution requirements for any data you redistribute. This repository does not vendor nflverse datasets.
