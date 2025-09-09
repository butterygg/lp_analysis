# Hedging Standard Deviation — bybit_funding / btc_usdt

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $32,337,860.78 |

### Standard Deviations by Percentile (period)

| Metric | p50 | p80 | p90 |
|---|---:|---:|---:|
| StdDev of hedging P&L | $13,058,080 | $26,885,081 | $46,974,775 |
| StdDev of metric change (%) | 50.07% | 103.09% | 180.12% |
| StdDev of metric change (abs) | 275.71 | 567.65 | 991.82 |

## Inputs & Diagnostics
- S→R pairs used: `418` (from `2024-06-25` to `2025-08-16`)
- Metric % Std (period): `102.73%` (std=`565.66`, median level=`550.65`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "bybit_funding",
  "profile": "btc_usdt",
  "exposure_to_metric": 47362.35,
  "average_total_exposure": 32337860.78,
  "delta_abs_percentiles": {
    "p50": 275.71,
    "p80": 567.65,
    "p90": 991.82
  },
  "metric_percent_std_pct": 102.73,
  "hedging_std_component": {
    "p50": 13058079.5,
    "p80": 26885081.14,
    "p90": 46974774.77
  },
  "debug": {
    "pair_count": 418,
    "first_S_date": "2024-06-25",
    "last_S_date": "2025-08-16",
    "metric_change_std": 565.66,
    "metric_level_median": 550.65,
    "metric_level_mean": 682.78
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/bybit_funding__btc_usdt__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/bybit_funding__btc_usdt__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  },
  "metric_percent_std_percentiles": {
    "p50": 50.07,
    "p80": 103.09,
    "p90": 180.12
  }
}
```
