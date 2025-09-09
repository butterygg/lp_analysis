# Hedging Standard Deviation — hyperliquid_funding / eth_usd

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $23,177,640.50 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $10,171,494 | $23,732,345 | $40,586,622 |
| StdDev of metric change (%) | 52.06% | 121.47% | 207.73% |
| StdDev of metric change (abs) | 560.95 | 1,308.83 | 2,238.33 |

## Inputs & Diagnostics
- S→R pairs used: `418` (from `2024-06-25` to `2025-08-16`)
- Metric % Std (period): `109.04%` (std=`1,174.92`, median level=`1,077.53`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "hyperliquid_funding",
  "profile": "eth_usd",
  "exposure_to_metric": 18132.53,
  "average_total_exposure": 23177640.5,
  "delta_abs_percentiles": {
    "p50": 560.95,
    "p80": 1308.83,
    "p90": 2238.33
  },
  "metric_percent_std_pct": 109.04,
  "hedging_std_component": {
    "p50": 10171493.96,
    "p80": 23732345.41,
    "p90": 40586621.71
  },
  "debug": {
    "pair_count": 418,
    "first_S_date": "2024-06-25",
    "last_S_date": "2025-08-16",
    "metric_change_std": 1174.92,
    "metric_level_median": 1077.53,
    "metric_level_mean": 1278.24
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/hyperliquid_funding__eth_usd__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/hyperliquid_funding__eth_usd__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 52.06,
    "p80": 121.47,
    "p90": 207.73
  }
}
```
