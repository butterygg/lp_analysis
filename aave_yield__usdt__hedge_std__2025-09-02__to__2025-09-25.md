# Hedging Standard Deviation — aave_yield / usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $20,161,878.34 |
| StdDev of hedging P&L (p50) | $3,266,259 |
| StdDev of hedging P&L (p80) | $7,240,713 |
| StdDev of hedging P&L (p90) | $10,165,672 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 19.56% |
| p80 | 43.37% |
| p90 | 60.89% |

## Inputs & Diagnostics
- S→R pairs used: `388` (from `2024-07-22` to `2025-08-13`)
- Metric % Std (period): `39.61%` (std=`150.44`, median level=`379.78`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdt",
  "exposure_to_metric": 43962.59,
  "average_total_exposure": 20161878.34,
  "delta_abs_percentiles": {
    "p50": 74.3,
    "p80": 164.7,
    "p90": 231.23
  },
  "metric_percentile_variance_pct": {
    "p50": 19.56,
    "p80": 43.37,
    "p90": 60.89
  },
  "hedging_std_component": {
    "p50": 3266258.82,
    "p80": 7240712.88,
    "p90": 10165671.97
  },
  "debug": {
    "pair_count": 388,
    "first_S_date": "2024-07-22",
    "last_S_date": "2025-08-13",
    "metric_change_std": 150.44,
    "metric_level_median": 379.78,
    "metric_level_mean": 458.61
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdt__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdt__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  }
}
```
