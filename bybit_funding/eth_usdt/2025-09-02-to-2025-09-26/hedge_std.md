# Hedging Standard Deviation — bybit_funding / eth_usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-26T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $230,549.82 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $87,469 | $196,654 | $390,494 |
| StdDev of metric change (%) | 50.67% | 113.96% | 226.32% |
| StdDev of metric change (abs) | 2.85 | 6.41 | 12.73 |

## Inputs & Diagnostics
- S→R pairs used: `416` (from `2024-07-01` to `2025-08-20`)
- Metric % Std (period): `130.76%` (std=`7.35`, median level=`5.62`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "bybit_funding",
  "profile": "eth_usdt",
  "exposure_to_metric": 30684.0,
  "average_total_exposure": 230549.82,
  "delta_abs_percentiles": {
    "p50": 2.85,
    "p80": 6.41,
    "p90": 12.73
  },
  "metric_percent_std_pct": 130.76,
  "hedging_std_component": {
    "p50": 87469.1,
    "p80": 196653.88,
    "p90": 390493.61
  },
  "debug": {
    "pair_count": 416,
    "first_S_date": "2024-07-01",
    "last_S_date": "2025-08-20",
    "metric_change_std": 7.35,
    "metric_level_median": 5.62,
    "metric_level_mean": 7.51
  },
  "report": {
    "path": "/home/pimania/dev/butter/mkt_reports/bybit_funding/eth_usdt/2025-09-02-to-2025-09-26/hedge_std.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/bybit_funding/eth_usdt/2025-09-02-to-2025-09-26/hedge_std.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-26T00:00:00+00:00",
    "period_days": 24
  },
  "metric_percent_std_percentiles": {
    "p50": 50.67,
    "p80": 113.96,
    "p90": 226.32
  }
}
```
