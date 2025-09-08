# Hedging Standard Deviation — eth_store_yield / eth

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Exposure to metric (window) | $1,039,892.81 |
| Average total exposure (metric × exposure/unit) | $328,299,480.76 |
| StdDev of hedging P&L (p50) | $6,006,765 |
| StdDev of hedging P&L (p80) | $12,597,503 |
| StdDev of hedging P&L (p90) | $19,125,756 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 1.83% |
| p80 | 3.83% |
| p90 | 5.82% |

## Inputs & Diagnostics
- S→R pairs used: `476` (from `2024-04-26` to `2025-08-14`)
- Metric % Std (period): `3.15%` (std=`9.95`, median level=`316.06`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "eth_store_yield",
  "profile": "eth",
  "exposure_to_metric": 1039892.81,
  "average_total_exposure": 328299480.76,
  "delta_abs_percentiles": {
    "p50": 5.78,
    "p80": 12.11,
    "p90": 18.39
  },
  "metric_percentile_variance_pct": {
    "p50": 1.83,
    "p80": 3.83,
    "p90": 5.82
  },
  "hedging_std_component": {
    "p50": 6006764.82,
    "p80": 12597503.3,
    "p90": 19125756.48
  },
  "debug": {
    "pair_count": 476,
    "first_S_date": "2024-04-26",
    "last_S_date": "2025-08-14",
    "metric_change_std": 9.95,
    "metric_level_median": 316.06,
    "metric_level_mean": 315.71
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/eth_store_yield__eth__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/eth_store_yield__eth__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
