# Hedging Standard Deviation — chain_fees / base

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $5,238,923.14 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $1,357,698 | $2,857,465 | $4,446,745 |
| StdDev of metric change (%) | 32.30% | 67.97% | 105.77% |
| StdDev of metric change (abs) | 59,030.35 | 124,237.59 | 193,336.74 |

## Inputs & Diagnostics
- S→R pairs used: `325` (from `2024-09-26` to `2025-08-16`)
- Metric % Std (period): `58.51%` (std=`106,947.91`, median level=`182,782.70`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "base",
  "exposure_to_metric": 23.0,
  "average_total_exposure": 5238923.14,
  "delta_abs_percentiles": {
    "p50": 59030.35,
    "p80": 124237.59,
    "p90": 193336.74
  },
  "metric_percent_std_pct": 58.51,
  "hedging_std_component": {
    "p50": 1357698.0,
    "p80": 2857464.6,
    "p90": 4446745.0
  },
  "debug": {
    "pair_count": 325,
    "first_S_date": "2024-09-26",
    "last_S_date": "2025-08-16",
    "metric_change_std": 106947.91,
    "metric_level_median": 182782.7,
    "metric_level_mean": 227779.27
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/chain_fees__base__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/chain_fees__base__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 32.3,
    "p80": 67.97,
    "p90": 105.77
  }
}
```
