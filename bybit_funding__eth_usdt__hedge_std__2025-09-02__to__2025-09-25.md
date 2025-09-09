# Hedging Standard Deviation — bybit_funding / eth_usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $22,273,737.43 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $8,505,779 | $18,426,056 | $37,059,603 |
| StdDev of metric change (%) | 50.95% | 110.38% | 222.01% |
| StdDev of metric change (abs) | 286.31 | 620.24 | 1,247.46 |

## Inputs & Diagnostics
- S→R pairs used: `418` (from `2024-06-25` to `2025-08-16`)
- Metric % Std (period): `129.71%` (std=`728.81`, median level=`561.89`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "bybit_funding",
  "profile": "eth_usdt",
  "exposure_to_metric": 29707.96,
  "average_total_exposure": 22273737.43,
  "delta_abs_percentiles": {
    "p50": 286.31,
    "p80": 620.24,
    "p90": 1247.46
  },
  "metric_percent_std_pct": 129.71,
  "hedging_std_component": {
    "p50": 8505778.64,
    "p80": 18426055.83,
    "p90": 37059602.57
  },
  "debug": {
    "pair_count": 418,
    "first_S_date": "2024-06-25",
    "last_S_date": "2025-08-16",
    "metric_change_std": 728.81,
    "metric_level_median": 561.89,
    "metric_level_mean": 749.76
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/bybit_funding__eth_usdt__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/bybit_funding__eth_usdt__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 50.95,
    "p80": 110.38,
    "p90": 222.01
  }
}
```
