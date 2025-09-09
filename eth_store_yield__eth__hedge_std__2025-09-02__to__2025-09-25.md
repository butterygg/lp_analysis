# Hedging Standard Deviation — eth_store_yield / eth

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $312,632,192.10 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $5,246,464 | $9,605,124 | $16,289,440 |
| StdDev of metric change (%) | 1.69% | 3.10% | 5.25% |
| StdDev of metric change (abs) | 5.26 | 9.64 | 16.34 |

## Inputs & Diagnostics
- S→R pairs used: `417` (from `2024-06-25` to `2025-08-15`)
- Metric % Std (period): `2.88%` (std=`8.97`, median level=`311.34`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "eth_store_yield",
  "profile": "eth",
  "exposure_to_metric": 996876.29,
  "average_total_exposure": 312632192.1,
  "delta_abs_percentiles": {
    "p50": 5.26,
    "p80": 9.64,
    "p90": 16.34
  },
  "metric_percent_std_pct": 2.88,
  "hedging_std_component": {
    "p50": 5246464.14,
    "p80": 9605124.45,
    "p90": 16289440.18
  },
  "debug": {
    "pair_count": 417,
    "first_S_date": "2024-06-25",
    "last_S_date": "2025-08-15",
    "metric_change_std": 8.97,
    "metric_level_median": 311.34,
    "metric_level_mean": 313.61
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/eth_store_yield__eth__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/eth_store_yield__eth__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 1.69,
    "p80": 3.1,
    "p90": 5.25
  }
}
```
