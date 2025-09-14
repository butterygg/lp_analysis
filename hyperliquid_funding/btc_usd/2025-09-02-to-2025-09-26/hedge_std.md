# Hedging Standard Deviation — hyperliquid_funding / btc_usd

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $363,398.90 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $128,356 | $257,964 | $397,600 |
| StdDev of metric change (%) | 39.80% | 79.98% | 123.27% |
| StdDev of metric change (abs) | 5.13 | 10.31 | 15.89 |

## Inputs & Diagnostics
- S→R pairs used: `416` (from `2024-07-01` to `2025-08-20`)
- Metric % Std (period): `67.35%` (std=`8.68`, median level=`12.89`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "hyperliquid_funding",
  "profile": "btc_usd",
  "exposure_to_metric": 25016.55,
  "average_total_exposure": 363398.9,
  "delta_abs_percentiles": {
    "p50": 5.13,
    "p80": 10.31,
    "p90": 15.89
  },
  "metric_percent_std_pct": 67.35,
  "hedging_std_component": {
    "p50": 128355.77,
    "p80": 257963.83,
    "p90": 397600.17
  },
  "debug": {
    "pair_count": 416,
    "first_S_date": "2024-07-01",
    "last_S_date": "2025-08-20",
    "metric_change_std": 8.68,
    "metric_level_median": 12.89,
    "metric_level_mean": 14.53
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/hyperliquid_funding/btc_usd/2025-09-02-to-2025-09-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/hyperliquid_funding/btc_usd/2025-09-02-to-2025-09-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-26T00:00:00+00:00",
    "period_days": 24
  },
  "metric_percent_std_percentiles": {
    "p50": 39.8,
    "p80": 79.98,
    "p90": 123.27
  }
}
```
