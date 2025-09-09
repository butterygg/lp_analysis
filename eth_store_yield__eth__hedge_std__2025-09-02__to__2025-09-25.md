# Hedging Standard Deviation — eth_store_yield / eth

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $314,569,141.65 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $5,931,234 | $12,157,983 | $18,462,246 |
| StdDev of metric change (%) | 1.88% | 3.86% | 5.87% |
| StdDev of metric change (abs) | 5.95 | 12.20 | 18.52 |

## Inputs & Diagnostics
- S→R pairs used: `478` (from `2024-04-26` to `2025-08-16`)
- Metric % Std (period): `3.18%` (std=`10.04`, median level=`315.75`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "eth_store_yield",
  "profile": "eth",
  "exposure_to_metric": 996876.29,
  "average_total_exposure": 314569141.65,
  "delta_abs_percentiles": {
    "p50": 5.95,
    "p80": 12.2,
    "p90": 18.52
  },
  "metric_percent_std_pct": 3.18,
  "hedging_std_component": {
    "p50": 5931233.53,
    "p80": 12157982.63,
    "p90": 18462246.01
  },
  "debug": {
    "pair_count": 478,
    "first_S_date": "2024-04-26",
    "last_S_date": "2025-08-16",
    "metric_change_std": 10.04,
    "metric_level_median": 315.75,
    "metric_level_mean": 315.55
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
    "p50": 1.88,
    "p80": 3.86,
    "p90": 5.87
  }
}
```
