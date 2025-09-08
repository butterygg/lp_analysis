# Hedging Standard Deviation — hyperliquid_funding / btc_usd

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $34,126,555.94 |
| StdDev of hedging P&L (p50) | $15,512,667 |
| StdDev of hedging P&L (p80) | $33,199,633 |
| StdDev of hedging P&L (p90) | $43,369,522 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 57.28% |
| p80 | 122.59% |
| p90 | 160.15% |

## Inputs & Diagnostics
- S→R pairs used: `434` (from `2024-06-08` to `2025-08-15`)
- Metric % Std (period): `101.94%` (std=`1,189.64`, median level=`1,167.01`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "hyperliquid_funding",
  "profile": "btc_usd",
  "exposure_to_metric": 23205.31,
  "average_total_exposure": 34126555.94,
  "delta_abs_percentiles": {
    "p50": 668.5,
    "p80": 1430.69,
    "p90": 1868.95
  },
  "metric_percentile_variance_pct": {
    "p50": 57.28,
    "p80": 122.59,
    "p90": 160.15
  },
  "hedging_std_component": {
    "p50": 15512666.84,
    "p80": 33199632.77,
    "p90": 43369521.9
  },
  "debug": {
    "pair_count": 434,
    "first_S_date": "2024-06-08",
    "last_S_date": "2025-08-15",
    "metric_change_std": 1189.64,
    "metric_level_median": 1167.01,
    "metric_level_mean": 1470.64
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/hyperliquid_funding__btc_usd__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/hyperliquid_funding__btc_usd__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  }
}
```
