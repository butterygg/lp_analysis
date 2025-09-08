# Hedging Variance Report

**KPI:** aave_yield  
**Profile:** usdt  
**Window (UTC):** 2025-08-01 → 2025-08-25  

## Inputs
- Exposure to metric: `45,874.005472`
- Fraction of portfolio exposed: `1.000000`
- Metric % Std over period: `40.7951%` (std=`155.077271`, median level=`380.137087`)

## Diagnostics
- S->R pairs used: `387`  (from `2024-07-22` to `2025-08-12`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdt",
  "exposure_to_metric": 45874.005472463905,
  "delta_squared_percentiles": {
    "p50": 5877.808888929907,
    "p80": 29411.983712486002,
    "p90": 58644.513907404355
  },
  "hedging_variance_component": {
    "p50": 12369404315604.371,
    "p80": 61895295532472.4,
    "p90": 123412944707841.86
  },
  "debug": {
    "pair_count": 387,
    "first_S_date": "2024-07-22",
    "last_S_date": "2025-08-12",
    "metric_change_std": 155.07727132972187,
    "metric_level_median": 380.1370869565191
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdt__hedge_variance__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdt__hedge_variance__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01",
    "end": "2025-08-25",
    "period_days": 24
  }
}
```
