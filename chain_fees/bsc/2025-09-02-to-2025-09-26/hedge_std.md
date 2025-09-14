# Hedging Standard Deviation — chain_fees / bsc

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $36,215.80 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $5,705 | $19,686 | $32,222 |
| StdDev of metric change (%) | 17.57% | 60.61% | 99.21% |
| StdDev of metric change (abs) | 86,816.75 | 299,586.15 | 490,363.43 |

## Inputs & Diagnostics
- S→R pairs used: `323` (from `2024-10-02` to `2025-08-20`)
- Metric % Std (period): `50.70%` (std=`250,570.83`, median level=`494,249.67`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "bsc",
  "exposure_to_metric": 0.07,
  "average_total_exposure": 36215.8,
  "delta_abs_percentiles": {
    "p50": 86816.75,
    "p80": 299586.15,
    "p90": 490363.43
  },
  "metric_percent_std_pct": 50.7,
  "hedging_std_component": {
    "p50": 5704.71,
    "p80": 19685.74,
    "p90": 32221.67
  },
  "debug": {
    "pair_count": 323,
    "first_S_date": "2024-10-02",
    "last_S_date": "2025-08-20",
    "metric_change_std": 250570.83,
    "metric_level_median": 494249.67,
    "metric_level_mean": 551147.87
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/chain_fees/bsc/2025-09-02-to-2025-09-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/chain_fees/bsc/2025-09-02-to-2025-09-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-26T00:00:00+00:00",
    "period_days": 24
  },
  "metric_percent_std_percentiles": {
    "p50": 17.57,
    "p80": 60.61,
    "p90": 99.21
  }
}
```
