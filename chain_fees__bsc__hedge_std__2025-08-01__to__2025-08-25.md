# Hedging Standard Deviation — chain_fees / bsc

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $1,290,230.01 |
| StdDev of hedging P&L (p50) | $316,788 |
| StdDev of hedging P&L (p80) | $806,088 |
| StdDev of hedging P&L (p90) | $1,545,780 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 29.39% |
| p80 | 74.79% |
| p90 | 143.41% |

## Inputs & Diagnostics
- S→R pairs used: `346` (from `2024-09-02` to `2025-08-13`)
- Metric % Std (period): `126.18%` (std=`56,666.29`, median level=`44,910.00`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "bsc",
  "exposure_to_metric": 24.0,
  "average_total_exposure": 1290230.01,
  "delta_abs_percentiles": {
    "p50": 13199.5,
    "p80": 33587.0,
    "p90": 64407.5
  },
  "metric_percentile_variance_pct": {
    "p50": 29.39,
    "p80": 74.79,
    "p90": 143.41
  },
  "hedging_std_component": {
    "p50": 316788.0,
    "p80": 806088.0,
    "p90": 1545780.0
  },
  "debug": {
    "pair_count": 346,
    "first_S_date": "2024-09-02",
    "last_S_date": "2025-08-13",
    "metric_change_std": 56666.29,
    "metric_level_median": 44910.0,
    "metric_level_mean": 53759.58
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/chain_fees__bsc__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/chain_fees__bsc__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
