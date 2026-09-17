# Experiment log

This file records promotion decisions for model experiments so later research does not silently reuse a rejected candidate or reinterpret a confirmation result after the fact.

## 2026-09-17 — Pace/scoring feature confirmation

### Operational baseline

The deployable baseline uses 78 pregame-only, market-independent features. Historical starter-QB fields reconstructed from the played game and realized temperature/wind/roof fields are blocked from fitting. The baseline 2022-2026 rolling-origin reference contains 1,155 completed test games.

### Development selection

The pace/scoring experiment split 72 prior-game engineered features into three blocks:

- `pace_volume` — 32 features
- `play_calling` — 16 features
- `scoring_efficiency` — 24 features

Selection was performed on 2019-2020 only. The selected candidate was `play_calling + scoring_efficiency`, adding 40 features to the 78-feature operational baseline. On the held-out 2021 internal validation season, that candidate improved all reported metrics versus the operational baseline, including margin MAE, total MAE, home-win Brier score, log loss, ECE, and exact-score NLL.

### One-shot 2022+ confirmation

The candidate was then frozen and compared once against the 78-feature baseline on identical rolling-origin seasons from 2022 through the completed portion of 2026. No alternate feature combination was tested on this confirmation surface.

Aggregate confirmation results:

| Metric | 78-feature baseline | 118-feature candidate | Candidate vs baseline |
| --- | ---: | ---: | ---: |
| Margin MAE | 9.940777 | 9.993741 | -0.5328% |
| Margin RMSE | 12.848858 | 12.892907 | -0.3428% |
| Total MAE | 10.593067 | 10.613785 | -0.1956% |
| Total RMSE | 13.517340 | 13.523402 | -0.0448% |
| Home-win Brier | 0.220572 | 0.221959 | -0.6288% |
| Home-win log loss | 0.631593 | 0.634723 | -0.4956% |
| Home-win ECE | 0.037776 | 0.026189 | +30.6730% |
| Exact-score NLL | 7.936101 | 7.944308 | -0.1034% |

Positive percentages mean lower error for the candidate. The precommitted primary metrics were margin MAE, total MAE, home-win Brier, and exact-score NLL. All four worsened. Their mean normalized loss ratio was approximately 1.003651, equivalent to a balanced deterioration of about 0.365%.

The result is not explained by the tiny 2026 sample. Across the four full confirmation seasons 2022-2025, the candidate still has worse aggregate margin MAE, total MAE, Brier score, log loss, and exact-score NLL. The candidate's clearest repeatable advantage is lower ECE, which is useful evidence for future probability-calibration research but does not justify accepting a model that is less accurate on the primary predictive targets.

### Decision

**Rejected for production.**

- Keep the 78-feature operational default unchanged.
- Do not retrain the production artifact with the 40 added pace/scoring features.
- Keep the experimental columns and confirmation command for reproducibility only.
- Do not test nearby pace/scoring subsets on the 2022+ confirmation surface; doing so would convert confirmation data back into a tuning set.
- Future feature work should return to pre-2022 development data and prioritize genuinely pregame sources such as timestamped starting-QB/injury/practice information, offensive-line and skill-position availability, and timestamped forecast weather.
- The ECE improvement may motivate a separate calibration experiment, but any calibration method must be selected exclusively inside the development era before later-era confirmation.
