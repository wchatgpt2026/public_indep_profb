# Model card

## Objective

Produce an independent pregame NFL probability model that outputs a coherent joint distribution over final home and away scores. The distribution can price any two-way moneyline, home spread, game total, or exact score without using sportsbook prices as predictive inputs.

## Architecture

1. **Leakage-safe team-form layer.** Play-by-play is aggregated to team-game efficiency statistics (EPA/play, success, dropback/rush EPA, explosive rate, turnovers, sacks, CPOE). Pregame values use exponentially weighted history shifted by one game. Schedule context contributes rest, neutral-site status and an internally maintained Elo rating.
2. **Experimental pace/scoring block.** The prepared dataset can additionally carry prior-game offensive snaps, drives, plays/drive, within-drive seconds/snap, no-huddle rate, early-down pass rate, red-zone EPA/success and scoring-drive rate, plus opponent-side counterparts. These columns remain non-default.
3. **Audit-only QB/weather context.** The dataset builder can derive historical QB state and retain nflverse temperature/wind/roof fields, but these are excluded from the deployable default model because the schedule QB identifiers are reconstructed from who actually played and the weather/roof fields describe realized game conditions rather than timestamped forecasts.
4. **Margin/total ensemble.** Margin and total are modeled separately using a regularized linear model, histogram gradient boosting and extremely randomized trees. Blend weights are learned on the chronologically latest 20% of each training window, then base learners are refit on the full window.
5. **Chronological out-of-fold calibration bank.** Expanding-window predictions are stored only for games the point model had not trained on. This bank is the basis of the score generator.
6. **Football-aware score distribution.** The generator finds historical games with similar ex-ante predicted margin/total, weights their realized score pairs by analog distance and recency, adds a weak empirically learned score prior, then maximum-entropy tilts the joint matrix until its home/away means match the current point forecast.

## Independence / leakage policy

Sportsbook moneylines, spread lines, total lines, odds and market-derived fields are excluded automatically and explicitly rejected if manually supplied as features. They can be used only outside the training feature matrix for benchmarking model prices after predictions are produced.

All team-form metrics, including the experimental pace/scoring block, are shifted by one game before exponential smoothing. A target game therefore cannot contribute its own play-by-play to those pregame rolling features.

The deployable default also blocks 14 historical context fields that are not valid as-of-prediction inputs from the current nflverse schedule source:

- starter-derived QB fields: `home_qb_epa`, `away_qb_epa`, `home_qb_cpoe`, `away_qb_cpoe`, `home_qb_experience`, `away_qb_experience`, `home_qb_known`, `away_qb_known`, `qb_epa_diff`, `qb_cpoe_diff`, `qb_experience_diff`
- realized game-condition fields: `temperature`, `wind_speed`, `indoors`

These fields may remain present in prepared datasets for audit or future research but are excluded automatically and rejected if manually passed to `NFLPredictor.fit`.

## Current operational baseline

The corrected default is 78 features on the current 2016-2026 dataset. A rolling-origin run over 2022-2026 contains 1,155 completed test games; the 2026 contribution is only 16 games and should be treated as a small partial-season sample.

As of 2026-09-17, the corrected operational benchmark is:

- margin MAE 9.9408 and RMSE 12.8489
- total MAE 10.5931 and RMSE 13.5173
- home-win Brier 0.22057 and log loss 0.63159
- exact-score NLL 7.93610
- versus the expanding-history naive baseline: +7.35% margin MAE, +7.47% margin RMSE, +2.67% total MAE, +1.82% total RMSE, and +10.72% home-win Brier improvement
- versus the evaluation-only nflverse market benchmark on matched games: -3.98% margin MAE, -3.75% total MAE, and -5.03% home-win Brier improvement, where negative means the market has lower error

This benchmark supersedes the earlier 92-feature results as the operational reference.

## Research-selection status

The earlier 2019-2021 pace/scoring experiments used the former 92-feature baseline that included postgame-derived QB/weather context. They are useful as research history but are not valid deployable validation results.

The development commands now use the same deployable pregame-only feature selector as the production model. `dev-compare`, `dev-ablate`, and `dev-target-split` therefore exclude the same postgame-context fields while continuing to strip market fields and refuse to enter the 2022+ confirmation era.

For active feature research, use 2019-2020 for feature-group selection and 2021 for internal validation. The already-inspected 2022+ benchmark is a secondary confirmation/reference surface rather than the primary tuning target.

## Intended evaluation

Use rolling-origin season/week holdouts. Primary metrics should include margin/total MAE and RMSE, moneyline Brier/log loss, exact-score negative log likelihood, calibration plots, interval coverage, and ATS/total calibration by line bucket. Compare against simple Elo and market benchmarks, but never feed the benchmark market price back into the independent model.

## Limitations

This repository establishes a research-grade architecture, not a claim of verified best-in-market predictive performance. Production quality depends on retraining cadence, data freshness, timestamped quarterback/injury information, forecast weather quality, roster changes, and disciplined walk-forward validation.

The current public-data build does not yet contain a timestamped pregame starter-QB feed, injury/practice participation snapshots, or forecast weather captured as-of prediction time. Those are high-value future extensions.

Repeated architecture changes after inspecting the same late-era benchmark can create research overfitting even when every individual split is technically chronological. The development/confirmation separation is intended to reduce that risk, not eliminate it entirely.
