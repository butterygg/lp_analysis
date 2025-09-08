# Hedging Standard Deviation — stable_deviation / usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $573,969,644.05 |
| StdDev of hedging P&L (p50) | $237,126,610 |
| StdDev of hedging P&L (p80) | $676,932,385 |
| StdDev of hedging P&L (p90) | $824,896,185 |

### Metric % Std (period)

| Metric | Value (%) |
|---|---:|
| Std/Median (over window) | 125.63% |

## Inputs & Diagnostics
- S→R pairs used: `325` (from `2024-09-21` to `2025-08-11`)
- Metric % Std (period): `125.63%` (std=`11.88`, median level=`9.46`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "stable_deviation",
  "profile": "usdt",
  "exposure_to_metric": 40055170.69,
  "average_total_exposure": 573969644.05,
  "delta_abs_percentiles": {
    "p50": 5.92,
    "p80": 16.9,
    "p90": 20.59
  },
  "metric_percent_std_pct": 125.63,
  "hedging_std_component": {
    "p50": 237126610.48,
    "p80": 676932384.66,
    "p90": 824896185.19
  },
  "debug": {
    "pair_count": 325,
    "first_S_date": "2024-09-21",
    "last_S_date": "2025-08-11",
    "metric_change_std": 11.88,
    "metric_level_median": 9.46,
    "metric_level_mean": 14.33
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/stable_deviation__usdt__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/stable_deviation__usdt__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  }
}
```
