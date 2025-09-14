# Hedging Standard Deviation — stable_deviation / usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $586,170,672.75 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $191,833,863 | $532,962,735 | $803,738,891 |
| StdDev of metric change (%) | 35.13% | 97.60% | 147.19% |
| StdDev of metric change (abs) | 4.69 | 13.03 | 19.65 |

## Inputs & Diagnostics
- S→R pairs used: `476` (from `2024-05-02` to `2025-08-20`)
- Metric % Std (period): `79.48%` (std=`10.61`, median level=`13.35`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "stable_deviation",
  "profile": "usdt",
  "exposure_to_metric": 40902742.55,
  "average_total_exposure": 586170672.75,
  "delta_abs_percentiles": {
    "p50": 4.69,
    "p80": 13.03,
    "p90": 19.65
  },
  "metric_percent_std_pct": 79.48,
  "hedging_std_component": {
    "p50": 191833862.54,
    "p80": 532962735.38,
    "p90": 803738891.04
  },
  "debug": {
    "pair_count": 476,
    "first_S_date": "2024-05-02",
    "last_S_date": "2025-08-20",
    "metric_change_std": 10.61,
    "metric_level_median": 13.35,
    "metric_level_mean": 14.33
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/stable_deviation/usdt/2025-09-02-to-2025-09-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/stable_deviation/usdt/2025-09-02-to-2025-09-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-26T00:00:00+00:00",
    "period_days": 24
  },
  "metric_percent_std_percentiles": {
    "p50": 35.13,
    "p80": 97.6,
    "p90": 147.19
  }
}
```
