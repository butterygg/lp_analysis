# Hedging Standard Deviation — aave_yield / usdc

**Window (UTC):** 2025-09-26T00:00:00+00:00 → 2025-10-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $25,225,983.41 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $4,056,607 | $15,665,638 | $18,412,547 |
| StdDev of metric change (%) | 20.61% | 79.94% | 93.84% |
| StdDev of metric change (abs) | 0.83 | 3.22 | 3.78 |

## Inputs & Diagnostics
- S→R pairs used: `374` (from `2024-08-09` to `2025-08-17`)
- Metric % Std (period): `57.45%` (std=`2.31`, median level=`4.03`)

## Raw Output (for programmatic use)
```json
{
  "metric": "aave_yield",
  "profile": "usdc",
  "exposure_to_metric": 4867042.54,
  "average_total_exposure": 25225983.41,
  "delta_abs_percentiles": {
    "p50": 0.83,
    "p80": 3.22,
    "p90": 3.78
  },
  "metric_percent_std_pct": 57.45,
  "hedging_std_component": {
    "p50": 4056606.95,
    "p80": 15665637.74,
    "p90": 18412547.06
  },
  "debug": {
    "pair_count": 374,
    "first_S_date": "2024-08-09",
    "last_S_date": "2025-08-17",
    "metric_change_std": 2.31,
    "metric_level_median": 4.03,
    "metric_level_mean": 5.18
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/aave_yield/usdc/2025-09-26-to-2025-10-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield/usdc/2025-09-26-to-2025-10-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-26T00:00:00+00:00",
    "end": "2025-10-26T00:00:00+00:00",
    "period_days": 30
  },
  "metric_percent_std_percentiles": {
    "p50": 20.61,
    "p80": 79.94,
    "p90": 93.84
  }
}
```
