# Hedging Standard Deviation — bybit_funding / btc_usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $344,721.32 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $139,758 | $283,719 | $498,119 |
| StdDev of metric change (%) | 49.92% | 101.27% | 177.66% |
| StdDev of metric change (abs) | 2.79 | 5.66 | 9.93 |

## Inputs & Diagnostics
- S→R pairs used: `416` (from `2024-07-01` to `2025-08-20`)
- Metric % Std (period): `102.02%` (std=`5.70`, median level=`5.59`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "bybit_funding",
  "profile": "btc_usdt",
  "exposure_to_metric": 50155.07,
  "average_total_exposure": 344721.32,
  "delta_abs_percentiles": {
    "p50": 2.79,
    "p80": 5.66,
    "p90": 9.93
  },
  "metric_percent_std_pct": 102.02,
  "hedging_std_component": {
    "p50": 139758.3,
    "p80": 283719.49,
    "p90": 498118.75
  },
  "debug": {
    "pair_count": 416,
    "first_S_date": "2024-07-01",
    "last_S_date": "2025-08-20",
    "metric_change_std": 5.7,
    "metric_level_median": 5.59,
    "metric_level_mean": 6.87
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/bybit_funding/btc_usdt/2025-09-02-to-2025-09-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/bybit_funding/btc_usdt/2025-09-02-to-2025-09-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-26T00:00:00+00:00",
    "period_days": 24
  },
  "metric_percent_std_percentiles": {
    "p50": 49.92,
    "p80": 101.27,
    "p90": 177.66
  }
}
```
