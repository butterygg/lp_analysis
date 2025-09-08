# Hedging Standard Deviation — hyperliquid_funding / eth_usd

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $24,172,798.12 |
| StdDev of hedging P&L (p50) | $16,626,320 |
| StdDev of hedging P&L (p80) | $41,197,009 |
| StdDev of hedging P&L (p90) | $56,179,600 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 82.45% |
| p80 | 204.30% |
| p90 | 278.60% |

## Inputs & Diagnostics
- S→R pairs used: `440` (from `2024-06-01` to `2025-08-14`)
- Metric % Std (period): `171.83%` (std=`1,877.17`, median level=`1,092.48`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "hyperliquid_funding",
  "profile": "eth_usd",
  "exposure_to_metric": 18458.07,
  "average_total_exposure": 24172798.12,
  "delta_abs_percentiles": {
    "p50": 900.76,
    "p80": 2231.92,
    "p90": 3043.63
  },
  "metric_percentile_variance_pct": {
    "p50": 82.45,
    "p80": 204.3,
    "p90": 278.6
  },
  "hedging_std_component": {
    "p50": 16626320.2,
    "p80": 41197008.85,
    "p90": 56179599.91
  },
  "debug": {
    "pair_count": 440,
    "first_S_date": "2024-06-01",
    "last_S_date": "2025-08-14",
    "metric_change_std": 1877.17,
    "metric_level_median": 1092.48,
    "metric_level_mean": 1309.61
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/hyperliquid_funding__eth_usd__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/hyperliquid_funding__eth_usd__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  }
}
```
