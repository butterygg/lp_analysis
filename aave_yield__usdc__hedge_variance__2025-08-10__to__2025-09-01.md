# Hedging Variance Report

**KPI:** aave_yield  
**Profile:** usdc  
**Window (UTC):** 2025-08-10 → 2025-09-01  

## Inputs
- Exposure to metric: `33,399.737675`
- Fraction of portfolio exposed: `1.000000`
- Metric % Std over period: `48.4037%` (std=`196.872672`, median level=`406.730391`)

## Diagnostics
- S->R pairs used: `389`  (from `2024-07-22` to `2025-08-14`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdc",
  "exposure_to_metric": 33399.737675164855,
  "delta_squared_percentiles": {
    "p50": 4875.0691837450695,
    "p80": 47439.67880915957,
    "p90": 95763.42267185201
  },
  "hedging_variance_component": {
    "p50": 5438346751659.232,
    "p80": 52920976795934.93,
    "p90": 106828165711313.56
  },
  "hedging_value_mean_variance": null,
  "debug": {
    "pair_count": 389,
    "first_S_date": "2024-07-22",
    "last_S_date": "2025-08-14",
    "metric_change_std": 196.87267154980836,
    "metric_level_median": 406.7303913043519
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdc__hedge_variance__2025-08-10__to__2025-09-01.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdc__hedge_variance__2025-08-10__to__2025-09-01.md"
  },
  "window": {
    "start": "2025-08-10",
    "end": "2025-09-01",
    "period_days": 22
  }
}
```
