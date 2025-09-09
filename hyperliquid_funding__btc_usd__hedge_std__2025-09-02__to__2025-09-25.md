# Hedging Standard Deviation — hyperliquid_funding / btc_usd

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $34,701,892.79 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $12,259,630 | $24,398,517 | $36,269,994 |
| StdDev of metric change (%) | 39.83% | 79.27% | 117.84% |
| StdDev of metric change (abs) | 515.65 | 1,026.22 | 1,525.55 |

## Inputs & Diagnostics
- S→R pairs used: `418` (from `2024-06-25` to `2025-08-16`)
- Metric % Std (period): `67.31%` (std=`871.38`, median level=`1,294.55`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "hyperliquid_funding",
  "profile": "btc_usd",
  "exposure_to_metric": 23775.05,
  "average_total_exposure": 34701892.79,
  "delta_abs_percentiles": {
    "p50": 515.65,
    "p80": 1026.22,
    "p90": 1525.55
  },
  "metric_percent_std_pct": 67.31,
  "hedging_std_component": {
    "p50": 12259630.08,
    "p80": 24398516.56,
    "p90": 36269993.94
  },
  "debug": {
    "pair_count": 418,
    "first_S_date": "2024-06-25",
    "last_S_date": "2025-08-16",
    "metric_change_std": 871.38,
    "metric_level_median": 1294.55,
    "metric_level_mean": 1459.59
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/hyperliquid_funding__btc_usd__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/hyperliquid_funding__btc_usd__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 39.83,
    "p80": 79.27,
    "p90": 117.84
  }
}
```
