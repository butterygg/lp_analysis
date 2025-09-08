# Hedging StdDev Report

**KPI:** stable_deviation  
**Profile:** usdt  
**Window (UTC):** 2025-08-01 → 2025-09-01  

## Inputs
- Exposure to metric: `53,987,403.973166`
- Fraction of portfolio exposed: `1.000000`
- Metric % Std over period: `130.2381%` (std=`12.320525`, median level=`9.460000`)

## Diagnostics
- S->R pairs used: `317`  (from `2024-09-21` to `2025-08-03`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "stable_deviation",
  "profile": "usdt",
  "exposure_to_metric": 53987403.97316628,
  "delta_abs_percentiles": {
    "p50": 5.020000000000024,
    "p80": 16.899999999999693,
    "p90": 19.82000000000039
  },
  "hedging_std_component": {
    "p50": 271016767.94529605,
    "p80": 912387127.1464936,
    "p90": 1070030346.7481768
  },
  "debug": {
    "pair_count": 317,
    "first_S_date": "2024-09-21",
    "last_S_date": "2025-08-03",
    "metric_change_std": 12.32052510500903,
    "metric_level_median": 9.460000000000024
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/stable_deviation__usdt__hedge_variance__2025-08-01__to__2025-09-01.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/stable_deviation__usdt__hedge_variance__2025-08-01__to__2025-09-01.md"
  },
  "window": {
    "start": "2025-08-01",
    "end": "2025-09-01",
    "period_days": 31
  }
}
```
