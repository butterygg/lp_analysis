# Hedging Variance Report

**KPI:** chain_fees  
**Profile:** base  
**Window (UTC):** 2025-09-01 → 2025-09-02  

## Inputs
- Exposure to metric: `1.000000`
- Fraction of portfolio exposed: `1.000000`
- Metric % Std over period: `61.7022%` (std=`103,699.799239`, median level=`168,065.000000`)

## Diagnostics
- S->R pairs used: `369`  (from `2024-09-02` to `2025-09-05`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "base",
  "exposure_to_metric": 1.0,
  "delta_squared_percentiles": {
    "p50": 1988089744.0,
    "p80": 11383636781.200018,
    "p90": 26790641866.399994
  },
  "hedging_variance_component": {
    "p50": 1988089744.0,
    "p80": 11383636781.200018,
    "p90": 26790641866.399994
  },
  "debug": {
    "pair_count": 369,
    "first_S_date": "2024-09-02",
    "last_S_date": "2025-09-05",
    "metric_change_std": 103699.79923907146,
    "metric_level_median": 168065.0
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/chain_fees__base__hedge_variance__2025-09-01__to__2025-09-02.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/chain_fees__base__hedge_variance__2025-09-01__to__2025-09-02.md"
  },
  "window": {
    "start": "2025-09-01",
    "end": "2025-09-02",
    "period_days": 1
  }
}
```
