# Weekly expanding-refit experiment

## Hypothesis

The production backtest currently fits one model from seasons before test season S and keeps those model parameters fixed for the entire test season. Pregame rolling features still update as games are completed, but the estimator itself does not learn from the current season until the following year.

This experiment tests one fixed operational change: refit the same 78-feature model at each NFL week boundary using all completed prior seasons plus completed games from earlier weeks of the current season.

No feature set changes, sportsbook inputs, postgame QB/weather context, or new hyperparameter search are involved.

## Leakage rule

For test season S and week W:

```text
train = all completed games from seasons < S
      + completed games from season S with week < W

test  = games from season S, week W
```

Games earlier in the same NFL week are intentionally excluded even if they were played on Thursday or Saturday. This makes the cutoff conservative and auditable.

Week 1 therefore uses exactly the same training history as the existing season-frozen backtest. From Week 2 onward, the weekly model can learn from the current season.

## Development protocol

The model remains locked to the 78 deployable pregame-only features.

- selection comparison: 2019-2020
- internal validation: 2021
- 2022+ remains outside this development run
- selection metrics: margin MAE, total MAE, home-win Brier, exact-score NLL

The precommitted development gate requires all of the following:

1. balanced normalized loss across the four selection metrics improves on 2019-2020;
2. the same balanced loss improves on 2021;
3. 2021 margin MAE does not worsen; and
4. 2021 total MAE does not worsen.

Only if the gate passes should the weekly-refit policy move to a later-era confirmation. Because 2022+ has been examined by prior experiments, any later comparison is secondary confirmation rather than a pristine selection surface.

## Run

```bash
python -m nflprob.weekly_refit_cli \
  --data data/games.parquet \
  --selection-start-season 2019 \
  --selection-end-season 2020 \
  --validation-season 2021 \
  --predictions-output artifacts/dev_weekly_refit_2021.csv
```

The normal production `train` and `backtest` commands are unchanged by this experiment.
