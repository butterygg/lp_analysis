# Hedging Standard Deviation — chain_fees / base

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $16,158.28 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $3,954 | $7,932 | $15,577 |
| StdDev of metric change (%) | 32.38% | 64.95% | 127.54% |
| StdDev of metric change (abs) | 60,179.83 | 120,719.14 | 237,051.74 |

## Inputs & Diagnostics
- S→R pairs used: `323` (from `2024-10-02` to `2025-08-20`)
- Metric % Std (period): `63.93%` (std=`118,816.70`, median level=`185,863.46`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "base",
  "exposure_to_metric": 0.07,
  "average_total_exposure": 16158.28,
  "delta_abs_percentiles": {
    "p50": 60179.83,
    "p80": 120719.14,
    "p90": 237051.74
  },
  "metric_percent_std_pct": 63.93,
  "hedging_std_component": {
    "p50": 3954.4,
    "p80": 7932.43,
    "p90": 15576.61
  },
  "debug": {
    "pair_count": 323,
    "first_S_date": "2024-10-02",
    "last_S_date": "2025-08-20",
    "metric_change_std": 118816.7,
    "metric_level_median": 185863.46,
    "metric_level_mean": 245903.71
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/chain_fees/base/2025-09-02-to-2025-09-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/chain_fees/base/2025-09-02-to-2025-09-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-26T00:00:00+00:00",
    "period_days": 24
  },
  "metric_percent_std_percentiles": {
    "p50": 32.38,
    "p80": 64.95,
    "p90": 127.54
  }
}
```
