# Hedging Standard Deviation — eth_store_yield / eth

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Exposure to metric (window) | $1,039,892.81 |
| StdDev of hedging P&L (p50) | $3,653,861 |
| StdDev of hedging P&L (p80) | $7,020,984 |
| StdDev of hedging P&L (p90) | $9,011,621 |

## Inputs & Diagnostics
- S→R pairs used: `317` (from `2024-09-27` to `2025-08-09`)
- Metric % Std (period): `1.42%` (std=`4.36`, median level=`308.05`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "eth_store_yield",
  "profile": "eth",
  "exposure_to_metric": 1039892.81,
  "delta_abs_percentiles": {
    "p50": 3.51,
    "p80": 6.75,
    "p90": 8.67
  },
  "hedging_std_component": {
    "p50": 3653861.12,
    "p80": 7020984.18,
    "p90": 9011621.28
  },
  "debug": {
    "pair_count": 317,
    "first_S_date": "2024-09-27",
    "last_S_date": "2025-08-09",
    "metric_change_std": 4.36,
    "metric_level_median": 308.05
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
