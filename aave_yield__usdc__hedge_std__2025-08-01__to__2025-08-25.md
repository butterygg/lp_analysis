# Hedging Standard Deviation — aave_yield / usdc

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $18,928,250.51 |
| StdDev of hedging P&L (p50) | $2,769,936 |
| StdDev of hedging P&L (p80) | $8,927,778 |
| StdDev of hedging P&L (p90) | $11,650,530 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 18.75% |
| p80 | 60.44% |
| p90 | 78.88% |

## Inputs & Diagnostics
- S→R pairs used: `384` (from `2024-07-27` to `2025-08-14`)
- Metric % Std (period): `50.07%` (std=`202.97`, median level=`405.38`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdc",
  "exposure_to_metric": 36436.08,
  "average_total_exposure": 18928250.51,
  "delta_abs_percentiles": {
    "p50": 76.02,
    "p80": 245.03,
    "p90": 319.75
  },
  "metric_percentile_variance_pct": {
    "p50": 18.75,
    "p80": 60.44,
    "p90": 78.88
  },
  "hedging_std_component": {
    "p50": 2769936.47,
    "p80": 8927778.19,
    "p90": 11650530.36
  },
  "debug": {
    "pair_count": 384,
    "first_S_date": "2024-07-27",
    "last_S_date": "2025-08-14",
    "metric_change_std": 202.97,
    "metric_level_median": 405.38,
    "metric_level_mean": 519.49
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdc__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdc__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
