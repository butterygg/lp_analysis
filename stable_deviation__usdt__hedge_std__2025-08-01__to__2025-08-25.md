# Hedging Standard Deviation — stable_deviation / usdt

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $600,110,308.34 |
| StdDev of hedging P&L (p50) | $253,496,985 |
| StdDev of hedging P&L (p80) | $706,364,227 |
| StdDev of hedging P&L (p90) | $863,645,209 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 64.11% |
| p80 | 178.65% |
| p90 | 218.42% |

## Inputs & Diagnostics
- S→R pairs used: `324` (from `2024-09-21` to `2025-08-10`)
- Metric % Std (period): `126.58%` (std=`11.97`, median level=`9.46`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "stable_deviation",
  "profile": "usdt",
  "exposure_to_metric": 41796699.85,
  "average_total_exposure": 600110308.34,
  "delta_abs_percentiles": {
    "p50": 6.07,
    "p80": 16.9,
    "p90": 20.66
  },
  "metric_percentile_variance_pct": {
    "p50": 64.11,
    "p80": 178.65,
    "p90": 218.42
  },
  "hedging_std_component": {
    "p50": 253496984.59,
    "p80": 706364227.47,
    "p90": 863645209.0
  },
  "debug": {
    "pair_count": 324,
    "first_S_date": "2024-09-21",
    "last_S_date": "2025-08-10",
    "metric_change_std": 11.97,
    "metric_level_median": 9.46,
    "metric_level_mean": 14.36
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/stable_deviation__usdt__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/stable_deviation__usdt__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
