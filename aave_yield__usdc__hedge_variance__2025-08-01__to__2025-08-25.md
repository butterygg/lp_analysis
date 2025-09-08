# Hedging Variance Report

**KPI:** aave_yield  
**Profile:** usdc  
**Window (UTC):** 2025-08-01 → 2025-08-25  

## Inputs
- Exposure to metric: `36,436.077464`
- Fraction of portfolio exposed: `1.000000`
- Metric % Std over period: `51.5981%` (std=`210.064651`, median level=`407.116826`)

## Diagnostics
- S->R pairs used: `387`  (from `2024-07-22` to `2025-08-12`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdc",
  "exposure_to_metric": 36436.077463816204,
  "delta_squared_percentiles": {
    "p50": 5908.876582351632,
    "p80": 56544.98555128167,
    "p90": 108379.80992546512
  },
  "hedging_variance_component": {
    "p50": 7844552113511.921,
    "p80": 75068429630032.03,
    "p90": 143883707023453.53
  },
  "debug": {
    "pair_count": 387,
    "first_S_date": "2024-07-22",
    "last_S_date": "2025-08-12",
    "metric_change_std": 210.06465053116,
    "metric_level_median": 407.11682608694974
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdc__hedge_variance__2025-08-01__to__2025-08-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdc__hedge_variance__2025-08-01__to__2025-08-25.md"
  },
  "window": {
    "start": "2025-08-01",
    "end": "2025-08-25",
    "period_days": 24
  }
}
```
