# Hedging Standard Deviation — eth_store_yield / eth

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $314,571,883.21 |
| StdDev of hedging P&L (p50) | $5,785,208 |
| StdDev of hedging P&L (p80) | $12,073,950 |
| StdDev of hedging P&L (p90) | $17,980,662 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 1.84% |
| p80 | 3.83% |
| p90 | 5.71% |

## Inputs & Diagnostics
- S→R pairs used: `477` (from `2024-04-26` to `2025-08-15`)
- Metric % Std (period): `3.10%` (std=`9.79`, median level=`316.05`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "eth_store_yield",
  "profile": "eth",
  "exposure_to_metric": 996563.94,
  "average_total_exposure": 314571883.21,
  "delta_abs_percentiles": {
    "p50": 5.81,
    "p80": 12.12,
    "p90": 18.04
  },
  "metric_percentile_variance_pct": {
    "p50": 1.84,
    "p80": 3.83,
    "p90": 5.71
  },
  "hedging_std_component": {
    "p50": 5785207.57,
    "p80": 12073950.17,
    "p90": 17980662.14
  },
  "debug": {
    "pair_count": 477,
    "first_S_date": "2024-04-26",
    "last_S_date": "2025-08-15",
    "metric_change_std": 9.79,
    "metric_level_median": 316.05,
    "metric_level_mean": 315.66
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/eth_store_yield__eth__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/eth_store_yield__eth__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  }
}
```
