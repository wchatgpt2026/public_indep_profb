# Pregame context audit

`nflprob audit-pregame-context` checks whether public player-context data can support a genuinely as-of pregame feature layer before any such features are added to the model.

Run the default T-24 audit with:

```bash
nflprob audit-pregame-context \
  --start-season 2016 \
  --end-season 2026 \
  --cutoff-hours 24
```

The audit uses nflverse schedules to construct each game's cutoff timestamp. It then checks two public context sources under different evidence rules.

- Injury reports are eligible only through 2024, because the current upstream nflverse injury feed stops after that season. An injury record counts as available only when its `date_modified` timestamp is at or before the game cutoff.
- Pre-2025 depth charts are reported only as week-level joinability evidence. They do not contain a record timestamp, so the audit never labels them strict fixed-cutoff features.
- From 2025 onward, depth charts contain an ISO8601 `dt` snapshot timestamp. The audit selects only the latest QB1 snapshot at or before the cutoff and reports QB1 coverage, GSIS-ID coverage, and snapshot staleness.

The command is research-only. It does not modify the prepared game parquet, the production feature selector, a fitted model, or any backtest result.

The next modeling step should occur only after reviewing the audit output. A defensible historical QB/availability experiment would need to use information that is genuinely known before the target game, such as a lagged prior-game QB identity plus timestamp-eligible injury information. Current-season timestamped depth charts can be used as an operational override only where their as-of coverage is demonstrated by this audit.
