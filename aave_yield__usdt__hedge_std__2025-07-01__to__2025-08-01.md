# Hedging Standard Deviation — aave_yield / usdt

**Window (UTC):** 2025-07-01T00:00:00+00:00 → 2025-08-01T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Exposure to metric (window) | $59,253.92 |
| Average total exposure (metric × exposure/unit) | $27,307,832.03 |
| StdDev of hedging P&L (p50) | $6,137,713 |
| StdDev of hedging P&L (p80) | $12,546,690 |
| StdDev of hedging P&L (p90) | $17,683,265 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 27.19% |
| p80 | 55.58% |
| p90 | 78.34% |

## Inputs & Diagnostics
- S→R pairs used: `380` (from `2024-07-22` to `2025-08-05`)
- Metric % Std (period): `48.47%` (std=`184.67`, median level=`380.95`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdt",
  "exposure_to_metric": 59253.92,
  "average_total_exposure": 27307832.03,
  "delta_abs_percentiles": {
    "p50": 103.58,
    "p80": 211.74,
    "p90": 298.43
  },
  "metric_percentile_variance_pct": {
    "p50": 27.19,
    "p80": 55.58,
    "p90": 78.34
  },
  "hedging_std_component": {
    "p50": 6137713.35,
    "p80": 12546689.62,
    "p90": 17683264.65
  },
  "debug": {
    "pair_count": 380,
    "first_S_date": "2024-07-22",
    "last_S_date": "2025-08-05",
    "metric_change_std": 184.67,
    "metric_level_median": 380.95,
    "metric_level_mean": 460.86
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdt__hedge_std__2025-07-01__to__2025-08-01.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdt__hedge_std__2025-07-01__to__2025-08-01.md"
  },
  "window": {
    "start": "2025-07-01T00:00:00+00:00",
    "end": "2025-08-01T00:00:00+00:00",
    "period_days": 31
  }
}
```
