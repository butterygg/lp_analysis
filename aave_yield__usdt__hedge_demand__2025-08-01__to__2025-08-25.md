# Hedging Demand Report

**KPI:** aave_yield  
**Profile:** usdt  
**Window (UTC):** 2025-08-01 → 2025-08-25  

## Inputs
- Exposure to metric: `702,999.97`
- Correlation (metric_exposure_correlation): `1.0000`
- Fraction of portfolio exposed: `1.0000`
- Monthly variance percentiles (capped): p50=`1,114.10`, p80=`3,510.77`, p90=`6,071.14`

## Hedging Demand
- p50: `783,212,911.74`  
- p80: `2,468,067,841.49`  
- p90: `4,268,013,170.01`

## Diagnostics
- Months included: `14`  (from `2024-08-01` to `2025-09-01`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdt",
  "exposure_to_metric": 702999.9652739125,
  "metric_monthly_variance": {
    "p50": 1114.100925221913,
    "p80": 3510.7652395592177,
    "p90": 6071.142789238878
  },
  "hedging_demand": {
    "p50": 783212911.7426387,
    "p80": 2468067841.4949894,
    "p90": 4268013170.007896
  },
  "hedging_value_mean_variance": null,
  "debug": {
    "monthly_variance_count": 14,
    "first_month": "2024-08-01",
    "last_month": "2025-09-01"
  }
}
```
