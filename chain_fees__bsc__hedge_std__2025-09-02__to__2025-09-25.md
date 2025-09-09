# Hedging Standard Deviation — chain_fees / bsc

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $1,266,420.99 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $189,558 | $710,943 | $1,175,999 |
| StdDev of metric change (%) | 16.66% | 62.49% | 103.37% |
| StdDev of metric change (abs) | 8,241.65 | 30,910.56 | 51,130.40 |

## Inputs & Diagnostics
- S→R pairs used: `325` (from `2024-09-26` to `2025-08-16`)
- Metric % Std (period): `52.32%` (std=`25,881.08`, median level=`49,464.87`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "bsc",
  "exposure_to_metric": 23.0,
  "average_total_exposure": 1266420.99,
  "delta_abs_percentiles": {
    "p50": 8241.65,
    "p80": 30910.56,
    "p90": 51130.4
  },
  "metric_percent_std_pct": 52.32,
  "hedging_std_component": {
    "p50": 189558.0,
    "p80": 710942.8,
    "p90": 1175999.2
  },
  "debug": {
    "pair_count": 325,
    "first_S_date": "2024-09-26",
    "last_S_date": "2025-08-16",
    "metric_change_std": 25881.08,
    "metric_level_median": 49464.87,
    "metric_level_mean": 55061.78
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/chain_fees__bsc__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/chain_fees__bsc__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 16.66,
    "p80": 62.49,
    "p90": 103.37
  }
}
```
