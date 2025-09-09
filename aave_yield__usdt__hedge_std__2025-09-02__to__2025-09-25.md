# Hedging Standard Deviation — aave_yield / usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $20,037,869.02 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $3,255,297 | $7,216,414 | $10,131,557 |
| StdDev of metric change (%) | 19.60% | 43.44% | 60.99% |
| StdDev of metric change (abs) | 74.30 | 164.70 | 231.23 |

## Inputs & Diagnostics
- S→R pairs used: `388` (from `2024-07-25` to `2025-08-16`)
- Metric % Std (period): `39.63%` (std=`150.25`, median level=`379.11`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdt",
  "exposure_to_metric": 43815.05,
  "average_total_exposure": 20037869.02,
  "delta_abs_percentiles": {
    "p50": 74.3,
    "p80": 164.7,
    "p90": 231.23
  },
  "metric_percent_std_pct": 39.63,
  "hedging_std_component": {
    "p50": 3255297.49,
    "p80": 7216413.57,
    "p90": 10131556.71
  },
  "debug": {
    "pair_count": 388,
    "first_S_date": "2024-07-25",
    "last_S_date": "2025-08-16",
    "metric_change_std": 150.25,
    "metric_level_median": 379.11,
    "metric_level_mean": 457.33
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdt__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdt__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 19.6,
    "p80": 43.44,
    "p90": 60.99
  }
}
```
