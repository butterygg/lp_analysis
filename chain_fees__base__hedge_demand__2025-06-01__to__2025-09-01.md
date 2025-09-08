# Hedging Demand Report

**KPI:** chain_fees  
**Profile:** base  
**Window (UTC):** 2025-06-01 → 2025-09-01  

## Inputs
- Exposure to metric: `1.00`
- Correlation (metric_exposure_correlation): `1.0000`
- Fraction of portfolio exposed: `1.0000`
- Change variance over period (capped): p50=`21,947,829,904.00`, p80=`131,101,365,436.00`, p90=`208,093,258,954.40`

## Hedging Demand
- p50: `21,947,829,904.00`  
- p80: `131,101,365,436.00`  
- p90: `208,093,258,954.40`

## Diagnostics
- S->R pairs used: `279`  (from `2024-01-07` to `2025-06-07`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "base",
  "exposure_to_metric": 1.0,
  "period_change_variance": {
    "p50": 21947829904.0,
    "p80": 131101365436.00002,
    "p90": 208093258954.4
  },
  "hedging_demand": {
    "p50": 21947829904.0,
    "p80": 131101365436.00002,
    "p90": 208093258954.4
  },
  "hedging_value_mean_variance": null,
  "debug": {
    "pair_count": 279,
    "first_S_date": "2024-01-07",
    "last_S_date": "2025-06-07"
  }
}
```
