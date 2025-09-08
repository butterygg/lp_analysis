# Hedging Standard Deviation — chain_fees / base

**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Exposure to metric (window) | $24.00 |
| Average total exposure (metric × exposure/unit) | $5,306,040.76 |
| StdDev of hedging P&L (p50) | $2,502,744 |
| StdDev of hedging P&L (p80) | $4,993,968 |
| StdDev of hedging P&L (p90) | $6,880,272 |

### Metric Variance (percentiles)

| Percentile | Variance (%) |
|---:|---:|
| p50 | 61.08% |
| p80 | 121.88% |
| p90 | 167.91% |

## Inputs & Diagnostics
- S→R pairs used: `346` (from `2024-09-02` to `2025-08-13`)
- Metric % Std (period): `108.16%` (std=`184,667.26`, median level=`170,728.50`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "base",
  "exposure_to_metric": 24.0,
  "average_total_exposure": 5306040.76,
  "delta_abs_percentiles": {
    "p50": 104281.0,
    "p80": 208082.0,
    "p90": 286678.0
  },
  "metric_percentile_variance_pct": {
    "p50": 61.08,
    "p80": 121.88,
    "p90": 167.91
  },
  "hedging_std_component": {
    "p50": 2502744.0,
    "p80": 4993968.0,
    "p90": 6880272.0
  },
  "debug": {
    "pair_count": 346,
    "first_S_date": "2024-09-02",
    "last_S_date": "2025-08-13",
    "metric_change_std": 184667.26,
    "metric_level_median": 170728.5,
    "metric_level_mean": 221085.03
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/chain_fees__base__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/chain_fees__base__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
