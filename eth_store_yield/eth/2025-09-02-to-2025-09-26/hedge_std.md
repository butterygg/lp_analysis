# Hedging Standard Deviation — eth_store_yield / eth

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $3,253,848.71 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $63,881 | $125,587 | $178,396 |
| StdDev of metric change (%) | 1.93% | 3.86% | 5.47% |
| StdDev of metric change (abs) | 0.06 | 0.12 | 0.17 |

## Inputs & Diagnostics
- S→R pairs used: `415` (from `2024-07-01` to `2025-08-19`)
- Metric % Std (period): `3.16%` (std=`0.10`, median level=`3.11`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "eth_store_yield",
  "profile": "eth",
  "exposure_to_metric": 1041214.89,
  "average_total_exposure": 3253848.71,
  "delta_abs_percentiles": {
    "p50": 0.06,
    "p80": 0.12,
    "p90": 0.17
  },
  "metric_percent_std_pct": 3.16,
  "hedging_std_component": {
    "p50": 63881.33,
    "p80": 125587.33,
    "p90": 178395.82
  },
  "debug": {
    "pair_count": 415,
    "first_S_date": "2024-07-01",
    "last_S_date": "2025-08-19",
    "metric_change_std": 0.1,
    "metric_level_median": 3.11,
    "metric_level_mean": 3.13
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/eth_store_yield/eth/2025-09-02-to-2025-09-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/eth_store_yield/eth/2025-09-02-to-2025-09-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-26T00:00:00+00:00",
    "period_days": 24
  },
  "metric_percent_std_percentiles": {
    "p50": 1.93,
    "p80": 3.86,
    "p90": 5.47
  }
}
```
