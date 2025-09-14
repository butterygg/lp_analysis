# Hedging Standard Deviation — aave_yield / usdc

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $198,783.46 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $27,717 | $93,328 | $125,105 |
| StdDev of metric change (%) | 17.77% | 59.98% | 80.22% |
| StdDev of metric change (abs) | 0.72 | 2.43 | 3.25 |

## Inputs & Diagnostics
- S→R pairs used: `386` (from `2024-07-31` to `2025-08-20`)
- Metric % Std (period): `51.06%` (std=`2.07`, median level=`4.05`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdc",
  "exposure_to_metric": 38461.39,
  "average_total_exposure": 198783.46,
  "delta_abs_percentiles": {
    "p50": 0.72,
    "p80": 2.43,
    "p90": 3.25
  },
  "metric_percent_std_pct": 51.06,
  "hedging_std_component": {
    "p50": 27717.42,
    "p80": 93327.79,
    "p90": 125105.28
  },
  "debug": {
    "pair_count": 386,
    "first_S_date": "2024-07-31",
    "last_S_date": "2025-08-20",
    "metric_change_std": 2.07,
    "metric_level_median": 4.05,
    "metric_level_mean": 5.17
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/aave_yield/usdc/2025-09-02-to-2025-09-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield/usdc/2025-09-02-to-2025-09-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-26T00:00:00+00:00",
    "period_days": 24
  },
  "metric_percent_std_percentiles": {
    "p50": 17.77,
    "p80": 59.98,
    "p90": 80.22
  }
}
```
