# Hedging Standard Deviation — hyperliquid_funding / eth_usd

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $25,092,745.77 |
| StdDev of hedging P&L (p50) | $17,141,521 |
| StdDev of hedging P&L (p80) | $41,588,604 |
| StdDev of hedging P&L (p90) | $58,557,431 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 81.62% |
| p80 | 198.01% |
| p90 | 278.81% |

## Inputs & Diagnostics
- S→R pairs used: `439` (from `2024-06-01` to `2025-08-13`)
- Metric % Std (period): `171.30%` (std=`1,869.59`, median level=`1,091.39`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "hyperliquid_funding",
  "profile": "eth_usd",
  "exposure_to_metric": 19244.13,
  "average_total_exposure": 25092745.77,
  "delta_abs_percentiles": {
    "p50": 890.74,
    "p80": 2161.11,
    "p90": 3042.87
  },
  "metric_percentile_variance_pct": {
    "p50": 81.62,
    "p80": 198.01,
    "p90": 278.81
  },
  "hedging_std_component": {
    "p50": 17141520.53,
    "p80": 41588603.55,
    "p90": 58557430.77
  },
  "debug": {
    "pair_count": 439,
    "first_S_date": "2024-06-01",
    "last_S_date": "2025-08-13",
    "metric_change_std": 1869.59,
    "metric_level_median": 1091.39,
    "metric_level_mean": 1303.92
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/hyperliquid_funding__eth_usd__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/hyperliquid_funding__eth_usd__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
