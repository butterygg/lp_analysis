# Hedging Standard Deviation — hyperliquid_funding / eth_usd

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Exposure to metric (window) | $19,121.78 |
| StdDev of hedging P&L (p50) | $17,032,537 |
| StdDev of hedging P&L (p80) | $41,324,189 |
| StdDev of hedging P&L (p90) | $58,185,131 |

## Inputs & Diagnostics
- S→R pairs used: `439` (from `2024-06-01` to `2025-08-13`)
- Metric % Std (period): `171.30%` (std=`1,869.59`, median level=`1,091.39`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "hyperliquid_funding",
  "profile": "eth_usd",
  "exposure_to_metric": 19121.78,
  "delta_abs_percentiles": {
    "p50": 890.74,
    "p80": 2161.11,
    "p90": 3042.87
  },
  "hedging_std_component": {
    "p50": 17032537.09,
    "p80": 41324188.92,
    "p90": 58185130.66
  },
  "debug": {
    "pair_count": 439,
    "first_S_date": "2024-06-01",
    "last_S_date": "2025-08-13",
    "metric_change_std": 1869.59,
    "metric_level_median": 1091.39
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
