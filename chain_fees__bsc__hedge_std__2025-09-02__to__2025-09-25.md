# Hedging Standard Deviation — chain_fees / bsc

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $1,235,884.19 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $313,375 | $763,536 | $1,508,170 |
| StdDev of metric change (%) | 30.34% | 73.91% | 145.99% |
| StdDev of metric change (abs) | 13,625.00 | 33,197.20 | 65,572.60 |

## Inputs & Diagnostics
- S→R pairs used: `347` (from `2024-09-02` to `2025-08-14`)
- Metric % Std (period): `127.74%` (std=`57,372.47`, median level=`44,915.00`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "bsc",
  "exposure_to_metric": 23.0,
  "average_total_exposure": 1235884.19,
  "delta_abs_percentiles": {
    "p50": 13625.0,
    "p80": 33197.2,
    "p90": 65572.6
  },
  "metric_percent_std_pct": 127.74,
  "hedging_std_component": {
    "p50": 313375.0,
    "p80": 763535.6,
    "p90": 1508169.8
  },
  "debug": {
    "pair_count": 347,
    "first_S_date": "2024-09-02",
    "last_S_date": "2025-08-14",
    "metric_change_std": 57372.47,
    "metric_level_median": 44915.0,
    "metric_level_mean": 53734.1
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
    "p50": 30.34,
    "p80": 73.91,
    "p90": 145.99
  }
}
```
