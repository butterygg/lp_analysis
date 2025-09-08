# Hedging Standard Deviation — hyperliquid_funding / btc_usd

**Window (UTC):** 2025-08-29T00:00:00+00:00 → 2025-09-05T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Exposure to metric (window) | $7,163.67 |
| StdDev of hedging P&L (p50) | $3,381,199.17 |
| StdDev of hedging P&L (p80) | $6,981,645.44 |
| StdDev of hedging P&L (p90) | $10,264,668.03 |

## Inputs & Diagnostics
- S→R pairs used: `450` (from `2024-06-08` to `2025-08-31`)
- Metric % Std (period): `71.76%` (std=`844.99`, median level=`1,177.50`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "hyperliquid_funding",
  "profile": "btc_usd",
  "exposure_to_metric": 7163.67,
  "delta_abs_percentiles": {
    "p50": 471.99,
    "p80": 974.59,
    "p90": 1432.88
  },
  "hedging_std_component": {
    "p50": 3381199.17,
    "p80": 6981645.44,
    "p90": 10264668.03
  },
  "debug": {
    "pair_count": 450,
    "first_S_date": "2024-06-08",
    "last_S_date": "2025-08-31",
    "metric_change_std": 844.99,
    "metric_level_median": 1177.5
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/hyperliquid_funding__btc_usd__hedge_std__2025-08-29__to__2025-09-05.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/hyperliquid_funding__btc_usd__hedge_std__2025-08-29__to__2025-09-05.md"
  },
  "window": {
    "start": "2025-08-29T00:00:00+00:00",
    "end": "2025-09-05T00:00:00+00:00",
    "period_days": 7
  }
}
```
