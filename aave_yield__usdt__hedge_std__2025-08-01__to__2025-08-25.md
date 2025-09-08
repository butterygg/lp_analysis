# Hedging Standard Deviation — aave_yield / usdt

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $21,049,315.01 |
| StdDev of hedging P&L (p50) | $3,517,016 |
| StdDev of hedging P&L (p80) | $7,867,306 |
| StdDev of hedging P&L (p90) | $11,109,071 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 20.17% |
| p80 | 45.11% |
| p90 | 63.70% |

## Inputs & Diagnostics
- S→R pairs used: `387` (from `2024-07-22` to `2025-08-12`)
- Metric % Std (period): `40.80%` (std=`155.08`, median level=`380.14`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdt",
  "exposure_to_metric": 45874.01,
  "average_total_exposure": 21049315.01,
  "delta_abs_percentiles": {
    "p50": 76.67,
    "p80": 171.5,
    "p90": 242.16
  },
  "metric_percentile_variance_pct": {
    "p50": 20.17,
    "p80": 45.11,
    "p90": 63.7
  },
  "hedging_std_component": {
    "p50": 3517016.39,
    "p80": 7867306.17,
    "p90": 11109070.56
  },
  "debug": {
    "pair_count": 387,
    "first_S_date": "2024-07-22",
    "last_S_date": "2025-08-12",
    "metric_change_std": 155.08,
    "metric_level_median": 380.14,
    "metric_level_mean": 458.85
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdt__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdt__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
