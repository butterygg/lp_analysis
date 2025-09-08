# Hedging Demand Report

**KPI:** aave_yield  
**Profile:** usdc  
**Window (UTC):** 2025-08-01 → 2025-09-01  

## Inputs
- Exposure to metric: `548,166.06`
- Correlation (metric_exposure_correlation): `1.0000`
- Fraction of portfolio exposed: `1.0000`
- Monthly variance percentiles (capped): p50=`568.17`, p80=`7,543.96`, p90=`8,528.80`

## Hedging Demand
- p50: `311,453,570.93`  
- p80: `4,135,340,239.58`  
- p90: `4,675,200,923.48`

## Diagnostics
- Months included: `14`  (from `2024-08-01` to `2025-09-01`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdc",
  "exposure_to_metric": 548166.062617498,
  "metric_monthly_variance": {
    "p50": 568.1737564051473,
    "p80": 7543.955238363516,
    "p90": 8528.804029115043
  },
  "hedging_demand": {
    "p50": 311453570.931203,
    "p80": 4135340239.5763774,
    "p90": 4675200923.476246
  },
  "hedging_value_mean_variance": null,
  "debug": {
    "monthly_variance_count": 14,
    "first_month": "2024-08-01",
    "last_month": "2025-09-01"
  }
}
```
