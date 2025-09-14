# Hedging Standard Deviation — hyperliquid_funding / eth_usd

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $288,880.81 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $134,836 | $300,599 | $510,665 |
| StdDev of metric change (%) | 54.99% | 122.64% | 208.25% |
| StdDev of metric change (abs) | 5.91 | 13.18 | 22.38 |

## Inputs & Diagnostics
- S→R pairs used: `416` (from `2024-07-01` to `2025-08-20`)
- Metric % Std (period): `110.39%` (std=`11.86`, median level=`10.75`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "hyperliquid_funding",
  "profile": "eth_usd",
  "exposure_to_metric": 22814.32,
  "average_total_exposure": 288880.81,
  "delta_abs_percentiles": {
    "p50": 5.91,
    "p80": 13.18,
    "p90": 22.38
  },
  "metric_percent_std_pct": 110.39,
  "hedging_std_component": {
    "p50": 134835.89,
    "p80": 300598.57,
    "p90": 510664.55
  },
  "debug": {
    "pair_count": 416,
    "first_S_date": "2024-07-01",
    "last_S_date": "2025-08-20",
    "metric_change_std": 11.86,
    "metric_level_median": 10.75,
    "metric_level_mean": 12.66
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/hyperliquid_funding/eth_usd/2025-09-02-to-2025-09-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/hyperliquid_funding/eth_usd/2025-09-02-to-2025-09-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-26T00:00:00+00:00",
    "period_days": 24
  },
  "metric_percent_std_percentiles": {
    "p50": 54.99,
    "p80": 122.64,
    "p90": 208.25
  }
}
```
