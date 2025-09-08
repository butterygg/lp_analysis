# Hedging Standard Deviation — aave_yield / usdt

**Window (UTC):** 2025-08-15T00:00:00+00:00 → 2025-08-22T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $6,130,040.22 |
| StdDev of hedging P&L (p50) | $334,953 |
| StdDev of hedging P&L (p80) | $787,768 |
| StdDev of hedging P&L (p90) | $1,144,386 |

### Metric % Std (period)

| Metric | Value (%) |
|---|---:|
| Std/Median (over window) | 15.14% |

## Inputs & Diagnostics
- S→R pairs used: `404` (from `2024-07-22` to `2025-08-29`)
- Metric % Std (period): `15.14%` (std=`57.77`, median level=`381.67`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdt",
  "exposure_to_metric": 13379.92,
  "average_total_exposure": 6130040.22,
  "delta_abs_percentiles": {
    "p50": 25.03,
    "p80": 58.88,
    "p90": 85.53
  },
  "metric_percent_std_pct": 15.14,
  "hedging_std_component": {
    "p50": 334953.16,
    "p80": 787767.94,
    "p90": 1144385.57
  },
  "debug": {
    "pair_count": 404,
    "first_S_date": "2024-07-22",
    "last_S_date": "2025-08-29",
    "metric_change_std": 57.77,
    "metric_level_median": 381.67,
    "metric_level_mean": 458.15
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdt__hedge_std__2025-08-15__to__2025-08-22.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdt__hedge_std__2025-08-15__to__2025-08-22.md"
  },
  "window": {
    "start": "2025-08-15T00:00:00+00:00",
    "end": "2025-08-22T00:00:00+00:00",
    "period_days": 7
  }
}
```
