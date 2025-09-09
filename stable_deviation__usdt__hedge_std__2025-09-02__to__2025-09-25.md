# Hedging Standard Deviation — stable_deviation / usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $567,505,245.69 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $188,638,021 | $524,083,884 | $797,186,690 |
| StdDev of metric change (%) | 36.76% | 102.12% | 155.33% |
| StdDev of metric change (abs) | 4.69 | 13.03 | 19.82 |

## Inputs & Diagnostics
- S→R pairs used: `478` (from `2024-04-26` to `2025-08-16`)
- Metric % Std (period): `83.44%` (std=`10.65`, median level=`12.76`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "stable_deviation",
  "profile": "usdt",
  "exposure_to_metric": 40221326.45,
  "average_total_exposure": 567505245.69,
  "delta_abs_percentiles": {
    "p50": 4.69,
    "p80": 13.03,
    "p90": 19.82
  },
  "metric_percent_std_pct": 83.44,
  "hedging_std_component": {
    "p50": 188638021.03,
    "p80": 524083883.58,
    "p90": 797186690.15
  },
  "debug": {
    "pair_count": 478,
    "first_S_date": "2024-04-26",
    "last_S_date": "2025-08-16",
    "metric_change_std": 10.65,
    "metric_level_median": 12.76,
    "metric_level_mean": 14.11
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/stable_deviation__usdt__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/stable_deviation__usdt__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 36.76,
    "p80": 102.12,
    "p90": 155.33
  }
}
```
