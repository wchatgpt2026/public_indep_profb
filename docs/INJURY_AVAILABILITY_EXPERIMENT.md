# T-24 injury / practice availability experiment

## Why this experiment exists

The corrected production model intentionally excludes historical schedule QB identifiers and realized weather because those fields are not valid as-of-prediction inputs. The public nflverse injury feed, by contrast, includes `date_modified` timestamps and showed approximately 99% team-game coverage at a strict 24-hour pregame cutoff from 2016 through 2024 in the repository audit.

This experiment asks whether that genuinely pregame player-availability information improves the 78-feature operational baseline.

## Leakage rule

For each team-game, a player injury record is eligible only when:

```text
date_modified <= kickoff - 24 hours
```

If more than one eligible revision exists for the same player, only the latest eligible revision is used. Records modified after the cutoff are ignored even if they describe the same game week.

The experiment does not use historical actual-starter QB IDs, realized game weather, sportsbook inputs, or pre-2025 depth-chart snapshots as predictive features.

## Candidate feature groups

Three interpretable groups are evaluated:

1. `practice_load`: players on the report, DNP count, limited count, and a practice-burden score (`DNP=2`, `limited=1`).
2. `game_status`: out, doubtful and questionable counts plus a status-burden score (`out=3`, `doubtful=2`, `questionable=1`).
3. `position_concentration`: practice and game-status burden concentrated at QB, offensive line, skill positions, and defense.

Each team metric is represented as home, away, home-minus-away and home-plus-away. The complete block therefore contains 64 candidate columns.

## Research protocol

The command evaluates all seven non-empty combinations of those three groups on 2019-2020, using the same four selection metrics as the earlier guarded ablation:

- margin MAE
- total MAE
- home-win Brier score
- exact-score negative log likelihood

The winning combination is then evaluated once on 2021. Seasons 2022+ are not used for selection or internal validation.

Run:

```bash
nflprob dev-injury-ablate \
  --data data/games.parquet \
  --selection-start-season 2019 \
  --selection-end-season 2020 \
  --validation-season 2021 \
  --cutoff-hours 24 \
  --predictions-output artifacts/dev_injury_ablation_2021.csv
```

The command downloads schedules and injury reports for the training/development seasons at runtime; the existing game parquet does not need to be rebuilt.

## Development result

The 2019-2020 selection winner was `game_status + position_concentration` (48 added features), with a balanced normalized improvement of about 0.81% across the four precommitted selection metrics.

On 2021 validation, that same block improved margin MAE/RMSE and home-win Brier/log loss, but worsened total MAE/RMSE, ECE, and exact-score NLL. It is therefore not accepted as a universal feature block.

The fixed follow-up hypothesis is target-specific: add the already-selected `game_status + position_concentration` block to the margin model only, while leaving the total model on the 78-feature production baseline. This is not a new feature search.

## One-shot 2022-2024 confirmation

The injury source ends after 2024, so the fixed target-specific candidate is confirmed once on 2022-2024. The precommitted acceptance rule requires all of the following:

- margin MAE improves;
- home-win Brier improves;
- exact-score NLL does not worsen; and
- total point predictions remain invariant within `1e-6`.

Run:

```bash
nflprob-confirm-injury-margin \
  --data data/games.parquet \
  --start-season 2022 \
  --end-season 2024 \
  --cutoff-hours 24 \
  --predictions-output artifacts/confirmation_injury_margin_2022_2024.csv
```

The command downloads schedules and injury reports through 2024 at runtime. Production defaults remain unchanged unless this fixed candidate clears the confirmation rule.

## Source limitation

The nflverse injury feed currently ends after the 2024 season. Even if the historical confirmation succeeds, operational deployment for 2025+ requires a current injury/practice source with equivalent timestamp semantics. The 2025+ timestamped depth-chart feed is useful for current QB1 context, but it is not a substitute for the missing injury/practice feed.
