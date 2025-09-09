# Hedging Standard Deviation — aave_yield / usdc

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $18,307,551.31 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $2,530,501 | $8,074,200 | $11,249,929 |
| StdDev of metric change (%) | 17.66% | 56.34% | 78.49% |
| StdDev of metric change (abs) | 71.67 | 228.68 | 318.63 |

## Inputs & Diagnostics
- S→R pairs used: `388` (from `2024-07-25` to `2025-08-16`)
- Metric % Std (period): `50.10%` (std=`203.37`, median level=`405.93`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdc",
  "exposure_to_metric": 35307.62,
  "average_total_exposure": 18307551.31,
  "delta_abs_percentiles": {
    "p50": 71.67,
    "p80": 228.68,
    "p90": 318.63
  },
  "metric_percent_std_pct": 50.1,
  "hedging_std_component": {
    "p50": 2530501.06,
    "p80": 8074200.27,
    "p90": 11249928.77
  },
  "debug": {
    "pair_count": 388,
    "first_S_date": "2024-07-25",
    "last_S_date": "2025-08-16",
    "metric_change_std": 203.37,
    "metric_level_median": 405.93,
    "metric_level_mean": 518.52
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdc__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdc__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 17.66,
    "p80": 56.34,
    "p90": 78.49
  }
}
```
