# Hedging Standard Deviation — bybit_funding / eth_usdt

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Exposure to metric (window) | $31,065.59 |
| Average total exposure (metric × exposure/unit) | $23,282,827.25 |
| StdDev of hedging P&L (p50) | $8,993,061 |
| StdDev of hedging P&L (p80) | $20,041,378 |
| StdDev of hedging P&L (p90) | $39,376,686 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 51.52% |
| p80 | 114.81% |
| p90 | 225.58% |

## Inputs & Diagnostics
- S→R pairs used: `417` (from `2024-06-20` to `2025-08-10`)
- Metric % Std (period): `133.13%` (std=`748.04`, median level=`561.89`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "bybit_funding",
  "profile": "eth_usdt",
  "exposure_to_metric": 31065.59,
  "average_total_exposure": 23282827.25,
  "delta_abs_percentiles": {
    "p50": 289.49,
    "p80": 645.13,
    "p90": 1267.53
  },
  "metric_percentile_variance_pct": {
    "p50": 51.52,
    "p80": 114.81,
    "p90": 225.58
  },
  "hedging_std_component": {
    "p50": 8993060.79,
    "p80": 20041378.3,
    "p90": 39376685.77
  },
  "debug": {
    "pair_count": 417,
    "first_S_date": "2024-06-20",
    "last_S_date": "2025-08-10",
    "metric_change_std": 748.04,
    "metric_level_median": 561.89,
    "metric_level_mean": 749.47
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/bybit_funding__eth_usdt__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/bybit_funding__eth_usdt__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
