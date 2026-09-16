# Model card

## Objective

Produce an independent pregame NFL probability model that outputs a coherent joint distribution over final home and away scores. The distribution can price any two-way moneyline, home spread, game total, or exact score without using sportsbook prices as predictive inputs.

## Architecture

1. **Leakage-safe nflverse feature layer.** Play-by-play is aggregated to team-game efficiency statistics (EPA/play, success, dropback/rush EPA, explosive rate, turnovers, sacks, CPOE). Pregame values use exponentially weighted history shifted by one game. Schedule context adds rest, weather/roof, neutral-site status and an internally maintained Elo rating.
2. **Quarterback state.** Prior-game QB EPA/CPOE is exponentially weighted, experience-shrunk, and looked up strictly before the target kickoff when starter IDs are available.
3. **Margin/total ensemble.** Margin and total are modeled separately using a regularized linear model, histogram gradient boosting and extremely randomized trees. Blend weights are learned on the chronologically latest 20% of each training window, then base learners are refit on the full window.
4. **Chronological out-of-fold point calibration.** Expanding-window predictions are stored only for games the point model had not trained on. Separate affine corrections for margin and total can adjust systematic level and regression-to-the-mean bias. Their shrinkage strength is chosen on a later slice of the training OOF bank and then refit on all OOF rows, so the target test season never participates in calibration or tuning.
5. **Football-aware score distribution.** The generator finds historical games with similar calibrated ex-ante predicted margin/total, weights their realized score pairs by analog distance and recency, adds a weak empirically learned score prior, then maximum-entropy tilts the joint matrix until its home/away means match the current calibrated point forecast.

## Independence / leakage policy

Sportsbook moneylines, spread lines, total lines, odds and market-derived fields are excluded automatically and explicitly rejected if manually supplied as features. They can be used only outside the training feature matrix for benchmarking model prices after predictions are produced.

Point calibration and calibration-strength selection use only chronological out-of-fold predictions produced inside the current training window. A model evaluated on season S therefore learns its point model, calibration parameters and score-distribution bank entirely from seasons earlier than S.

## Intended evaluation

Use rolling-origin season/week holdouts. Primary metrics should include margin/total MAE and RMSE, moneyline Brier/log loss, exact-score negative log likelihood, calibration plots, interval coverage, and ATS/total calibration by line bucket. Compare against simple Elo and closing-market benchmarks, but never feed the benchmark market price back into the independent model.

## Limitations

This repository establishes a strong research-grade architecture, not a claim of verified best-in-market predictive performance. Production quality depends on retraining cadence, data freshness, quarterback/injury information, weather quality, roster changes, and disciplined walk-forward validation. Injury/practice participation and timestamped forecast weather remain important extensions. Repeatedly changing architecture after inspecting the same benchmark seasons can itself create research overfitting, so future model selection should increasingly rely on nested historical tuning and newly arriving games for confirmation.
