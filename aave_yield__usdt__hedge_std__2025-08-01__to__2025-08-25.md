# Hedging StdDev Report

**KPI:** aave_yield  
**Profile:** usdt  
**Window (UTC):** 2025-08-01T00:00:00+00:00 → 2025-08-25T00:00:00+00:00  

## Inputs
- Exposure to metric: `45,874.01`
- Fraction of portfolio exposed: `1.00`
- Metric % Std over period: `40.80%` (std=`155.08`, median level=`380.14`)

## Diagnostics
- S->R pairs used: `387`  (from `2024-07-22` to `2025-08-12`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdt",
  "exposure_to_metric": 45874.01,
  "delta_abs_percentiles": {
    "p50": 76.67,
    "p80": 171.5,
    "p90": 242.16
  },
  "hedging_std_component": {
    "p50": 3517016.39,
    "p80": 7867306.17,
    "p90": 11109070.56
  },
  "debug": {
    "pair_count": 387,
    "first_S_date": "2024-07-22",
    "last_S_date": "2025-08-12",
    "metric_change_std": 155.08,
    "metric_level_median": 380.14
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdt__hedge_std__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdt__hedge_std__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01T00:00:00+00:00",
    "end": "2025-08-25T00:00:00+00:00",
    "period_days": 24
  }
}
```
