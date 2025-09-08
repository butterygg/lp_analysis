# Hedging Standard Deviation — hyperliquid_funding / btc_usd

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $35,660,489.72 |
| StdDev of hedging P&L (p50) | $16,105,521 |
| StdDev of hedging P&L (p80) | $36,017,387 |
| StdDev of hedging P&L (p90) | $45,025,932 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 57.04% |
| p80 | 127.56% |
| p90 | 159.47% |

## Inputs & Diagnostics
- S→R pairs used: `433` (from `2024-06-08` to `2025-08-14`)
- Metric % Std (period): `102.81%` (std=`1,196.68`, median level=`1,163.95`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "hyperliquid_funding",
  "profile": "btc_usd",
  "exposure_to_metric": 24257.95,
  "average_total_exposure": 35660489.72,
  "delta_abs_percentiles": {
    "p50": 663.93,
    "p80": 1484.77,
    "p90": 1856.13
  },
  "metric_percentile_variance_pct": {
    "p50": 57.04,
    "p80": 127.56,
    "p90": 159.47
  },
  "hedging_std_component": {
    "p50": 16105520.96,
    "p80": 36017386.56,
    "p90": 45025932.34
  },
  "debug": {
    "pair_count": 433,
    "first_S_date": "2024-06-08",
    "last_S_date": "2025-08-14",
    "metric_change_std": 1196.68,
    "metric_level_median": 1163.95,
    "metric_level_mean": 1470.05
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/hyperliquid_funding__btc_usd__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/hyperliquid_funding__btc_usd__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
