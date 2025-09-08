# Hedging Standard Deviation — bybit_funding / eth_usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $22,320,815.81 |
| StdDev of hedging P&L (p50) | $8,609,680 |
| StdDev of hedging P&L (p80) | $18,465,274 |
| StdDev of hedging P&L (p90) | $37,138,480 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 51.47% |
| p80 | 110.38% |
| p90 | 222.01% |

## Inputs & Diagnostics
- S→R pairs used: `418` (from `2024-06-20` to `2025-08-11`)
- Metric % Std (period): `129.88%` (std=`729.80`, median level=`561.89`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "bybit_funding",
  "profile": "eth_usdt",
  "exposure_to_metric": 29771.19,
  "average_total_exposure": 22320815.81,
  "delta_abs_percentiles": {
    "p50": 289.2,
    "p80": 620.24,
    "p90": 1247.46
  },
  "metric_percentile_variance_pct": {
    "p50": 51.47,
    "p80": 110.38,
    "p90": 222.01
  },
  "hedging_std_component": {
    "p50": 8609680.36,
    "p80": 18465273.93,
    "p90": 37138480.39
  },
  "debug": {
    "pair_count": 418,
    "first_S_date": "2024-06-20",
    "last_S_date": "2025-08-11",
    "metric_change_std": 729.8,
    "metric_level_median": 561.89,
    "metric_level_mean": 749.75
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/bybit_funding__eth_usdt__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/bybit_funding__eth_usdt__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  }
}
```
