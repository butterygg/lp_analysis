# Hedging Standard Deviation — aave_yield / usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $205,272.58 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $34,623 | $76,565 | $111,872 |
| StdDev of metric change (%) | 20.33% | 44.89% | 65.75% |
| StdDev of metric change (abs) | 0.77 | 1.70 | 2.49 |

## Inputs & Diagnostics
- S→R pairs used: `386` (from `2024-07-31` to `2025-08-20`)
- Metric % Std (period): `40.35%` (std=`1.53`, median level=`3.79`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdt",
  "exposure_to_metric": 45010.68,
  "average_total_exposure": 205272.58,
  "delta_abs_percentiles": {
    "p50": 0.77,
    "p80": 1.7,
    "p90": 2.49
  },
  "metric_percent_std_pct": 40.35,
  "hedging_std_component": {
    "p50": 34623.05,
    "p80": 76564.79,
    "p90": 111871.88
  },
  "debug": {
    "pair_count": 386,
    "first_S_date": "2024-07-31",
    "last_S_date": "2025-08-20",
    "metric_change_std": 1.53,
    "metric_level_median": 3.79,
    "metric_level_mean": 4.56
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/aave_yield/usdt/2025-09-02-to-2025-09-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield/usdt/2025-09-02-to-2025-09-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-26T00:00:00+00:00",
    "period_days": 24
  },
  "metric_percent_std_percentiles": {
    "p50": 20.33,
    "p80": 44.89,
    "p90": 65.75
  }
}
```
