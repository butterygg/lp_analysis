# Hedging Standard Deviation — bybit_funding / btc_usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $32,214,484.85 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $13,261,095 | $27,841,149 | $46,872,240 |
| StdDev of metric change (%) | 50.96% | 106.99% | 180.12% |
| StdDev of metric change (abs) | 280.60 | 589.12 | 991.82 |

## Inputs & Diagnostics
- S→R pairs used: `418` (from `2024-06-20` to `2025-08-11`)
- Metric % Std (period): `103.62%` (std=`570.58`, median level=`550.65`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "bybit_funding",
  "profile": "btc_usdt",
  "exposure_to_metric": 47258.97,
  "average_total_exposure": 32214484.85,
  "delta_abs_percentiles": {
    "p50": 280.6,
    "p80": 589.12,
    "p90": 991.82
  },
  "metric_percent_std_pct": 103.62,
  "hedging_std_component": {
    "p50": 13261095.43,
    "p80": 27841149.35,
    "p90": 46872240.07
  },
  "debug": {
    "pair_count": 418,
    "first_S_date": "2024-06-20",
    "last_S_date": "2025-08-11",
    "metric_change_std": 570.58,
    "metric_level_median": 550.65,
    "metric_level_mean": 681.66
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/bybit_funding__btc_usdt__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/bybit_funding__btc_usdt__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 50.96,
    "p80": 106.99,
    "p90": 180.12
  }
}
```
