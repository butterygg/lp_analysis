# Hedging Standard Deviation — aave_yield / usdc

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $18,129,041.15 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $2,541,777 | $8,186,889 | $10,814,157 |
| StdDev of metric change (%) | 17.96% | 57.84% | 76.41% |
| StdDev of metric change (abs) | 72.79 | 234.46 | 309.70 |

## Inputs & Diagnostics
- S→R pairs used: `385` (from `2024-07-27` to `2025-08-15`)
- Metric % Std (period): `48.45%` (std=`196.37`, median level=`405.34`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdc",
  "exposure_to_metric": 34917.91,
  "average_total_exposure": 18129041.15,
  "delta_abs_percentiles": {
    "p50": 72.79,
    "p80": 234.46,
    "p90": 309.7
  },
  "metric_percent_std_pct": 48.45,
  "hedging_std_component": {
    "p50": 2541776.56,
    "p80": 8186888.87,
    "p90": 10814156.55
  },
  "debug": {
    "pair_count": 385,
    "first_S_date": "2024-07-27",
    "last_S_date": "2025-08-15",
    "metric_change_std": 196.37,
    "metric_level_median": 405.34,
    "metric_level_mean": 519.19
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
    "p50": 17.96,
    "p80": 57.84,
    "p90": 76.41
  }
}
```
