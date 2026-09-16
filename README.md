# public_indep_profb

Independent, research-grade NFL forecasting package that produces a **full joint final-score distribution** and can therefore price any moneyline, spread, total, or exact score from one internally coherent probability surface.

The design is intentionally market-independent: sportsbook lines and odds are blocked from the predictive feature matrix. Public nflverse play-by-play and schedule data provide the default inputs.

## What is implemented

- Leakage-safe pregame features from EPA/play, success rate, dropback/rush EPA, explosives, turnovers, sacks, CPOE, points, rest, weather/roof and an internal Elo rating.
- An experimental, non-default pace/scoring feature block with prior-game snaps, drives, plays/drive, within-drive seconds/snap, no-huddle rate, early-down pass rate, red-zone EPA/success and scoring-drive rate, plus opponent-side counterparts.
- Exponentially weighted team form with current-game data shifted out of every pregame row.
- A quarterback state layer using prior-game QB EPA/CPOE, experience shrinkage, and strictly pre-kickoff lookup when starter IDs are available.
- Separate **margin** and **total** ensembles: ridge + histogram gradient boosting + Extra Trees, with chronological validation used to learn non-negative blend weights.
- Expanding-window out-of-fold forecasts for distribution calibration.
- A discrete **joint score distribution** based on ex-ante analog games plus recency weighting and maximum-entropy tilting to the point model's target home/away means.
- Fair moneyline, spread and total pricing with push handling and fair decimal/American odds.
- Season-by-season rolling-origin backtesting with calibration, naive historical baselines, and evaluation-only nflverse market benchmarks.
- Guarded development workflows that strip market fields and keep 2022+ out of feature selection.
- CLI, serialization, holdout evaluation, unit tests and GitHub Actions CI.

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for assumptions and limitations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[data,dev]'
```

## End-to-end run

Build a dataset. A practical starting window is 2016 through the current season; older seasons can be added if desired.

```bash
nflprob build-data \
  --start-season 2016 \
  --end-season 2026 \
  --output data/games.parquet
```

The prepared dataset can contain experimental columns without changing normal model behavior. `nflprob train` and `nflprob backtest` continue to use the accepted legacy feature set unless an experiment is explicitly promoted in code.

### Research candidate selection

Use the reserved pre-2022 development window to compare the accepted feature set against the full experimental pace/scoring block:

```bash
nflprob dev-compare \
  --data data/games.parquet \
  --start-season 2019 \
  --end-season 2021 \
  --predictions-output artifacts/dev_candidate_predictions.csv
```

For a narrower selection pass, split the development era again. `dev-ablate` evaluates all seven non-empty combinations of three interpretable feature groups on 2019-2020 only, chooses the lowest balanced normalized loss across margin MAE, total MAE, home-win Brier score and exact-score NLL, and then evaluates that winner once on 2021:

```bash
nflprob dev-ablate \
  --data data/games.parquet \
  --selection-start-season 2019 \
  --selection-end-season 2020 \
  --validation-season 2021 \
  --predictions-output artifacts/dev_ablation_winner_2021.csv
```

The three groups are **pace/volume** (snaps, drives, plays/drive, within-drive tempo), **play calling** (no-huddle and early-down pass rate), and **scoring efficiency** (red-zone EPA/success and scoring-drive rate), each with opponent-side counterparts. Both development commands remove sportsbook columns before fitting or scoring and refuse to enter the reserved 2022+ confirmation era.

Train the accepted/default artifact:

```bash
nflprob train \
  --data data/games.parquet \
  --model artifacts/nfl_model.joblib
```

### Confirmation benchmark

After a candidate has been selected on the development window, run the later rolling-origin confirmation benchmark once. Each test season is predicted by a fresh model trained only on completed games from earlier seasons:

```bash
nflprob backtest \
  --data data/games.parquet \
  --start-season 2022 \
  --end-season 2026 \
  --predictions-output artifacts/walk_forward_predictions.csv
```

The report includes aggregate and season-level margin/total MAE and RMSE, home-moneyline Brier score, binary log loss, expected calibration error, exact-score negative log likelihood, and comparison against an expanding-history constant baseline. When the nflverse schedule supplies market fields, the same report also compares the independent model with the posted spread, total and **no-vig** moneyline implied probability on the exact same games. Market fields are retained solely after prediction for evaluation and remain blocked from model training.

The predictions CSV is suitable for deeper calibration plots, residual analysis and line-by-line comparison.

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

That guarantees internally consistent prices: a moneyline, -3.5 spread and 47.5 total are all integrations of the **same** joint score matrix rather than separate classifiers that can contradict one another.

## Highest-value next extensions

1. Use nested development-era ablation to keep only robust pace/scoring signal, then confirm the selected configuration once on 2022+.
2. Timestamped injury/practice participation plus an explicit late-QB-scratch override path.
3. Offensive-line continuity and skill-position availability features.
4. Stadium-specific forecast weather captured as-of prediction time.
5. Nested walk-forward hyperparameter and score-distribution tuning inside development-era data only.
6. Automated weekly data refresh/retrain/publish workflow after the research protocol is accepted.

## Data attribution

Default data access is via the nflverse ecosystem. Review the upstream nflverse licenses and attribution requirements for any data you redistribute. This repository does not vendor nflverse datasets.
