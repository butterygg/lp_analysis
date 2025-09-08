# Hedging StdDev Report

**KPI:** aave_yield  
**Profile:** usdt  
**Window (UTC):** 2025-08-10T00:00:00+00:00 → 2025-09-05T00:00:00+00:00  

## Inputs
- Exposure to metric: `49,696.84`
- Fraction of portfolio exposed: `1.00`
- Metric % Std over period: `43.03%` (std=`163.73`, median level=`380.52`)

## Diagnostics
- S->R pairs used: `385`  (from `2024-07-22` to `2025-08-10`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdt",
  "exposure_to_metric": 49696.84,
  "delta_abs_percentiles": {
    "p50": 87.3,
    "p80": 180.65,
    "p90": 266.33
  },
  "hedging_std_component": {
    "p50": 4338445.48,
    "p80": 8977788.9,
    "p90": 13235954.96
  },
  "debug": {
    "pair_count": 385,
    "first_S_date": "2024-07-22",
    "last_S_date": "2025-08-10",
    "metric_change_std": 163.73,
    "metric_level_median": 380.52
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdt__hedge_std__2025-08-10__to__2025-09-05.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdt__hedge_std__2025-08-10__to__2025-09-05.md"
  },
  "window": {
    "start": "2025-08-10T00:00:00+00:00",
    "end": "2025-09-05T00:00:00+00:00",
    "period_days": 26
  }
}
```
