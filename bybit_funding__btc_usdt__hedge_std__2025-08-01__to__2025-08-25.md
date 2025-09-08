# Hedging Standard Deviation — bybit_funding / btc_usdt

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $33,590,442.56 |
| StdDev of hedging P&L (p50) | $14,160,286 |
| StdDev of hedging P&L (p80) | $29,665,370 |
| StdDev of hedging P&L (p90) | $50,257,364 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 52.20% |
| p80 | 109.36% |
| p90 | 185.27% |

## Inputs & Diagnostics
- S→R pairs used: `417` (from `2024-06-20` to `2025-08-10`)
- Metric % Std (period): `106.17%` (std=`584.04`, median level=`550.08`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "bybit_funding",
  "profile": "btc_usdt",
  "exposure_to_metric": 49313.7,
  "average_total_exposure": 33590442.56,
  "delta_abs_percentiles": {
    "p50": 287.15,
    "p80": 601.56,
    "p90": 1019.14
  },
  "metric_percentile_variance_pct": {
    "p50": 52.2,
    "p80": 109.36,
    "p90": 185.27
  },
  "hedging_std_component": {
    "p50": 14160286.3,
    "p80": 29665370.16,
    "p90": 50257363.65
  },
  "debug": {
    "pair_count": 417,
    "first_S_date": "2024-06-20",
    "last_S_date": "2025-08-10",
    "metric_change_std": 584.04,
    "metric_level_median": 550.08,
    "metric_level_mean": 681.16
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/bybit_funding__btc_usdt__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/bybit_funding__btc_usdt__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
