# Hedging Demand Report

**KPI:** aave_yield  
**Profile:** usdc  
**Window (UTC):** 2025-06-01 → 2025-09-01  

## Inputs
- Exposure to metric: `548,166.06`
- Correlation (metric_exposure_correlation): `1.0000`
- Fraction of portfolio exposed: `1.0000`
- Change variance over period (capped): p50=`40,168.91`, p80=`289,961.14`, p90=`560,985.14`

## Hedging Demand
- p50: `22,019,234,779.49`  
- p80: `158,946,853,892.21`  
- p90: `307,513,017,942.04`

## Diagnostics
- S->R pairs used: `320`  (from `2023-08-28` to `2025-06-06`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdc",
  "exposure_to_metric": 548166.062617498,
  "metric_monthly_variance": {
    "p50": 40168.91281876023,
    "p80": 289961.13537790545,
    "p90": 560985.1446725185
  },
  "hedging_demand": {
    "p50": 22019234779.48534,
    "p80": 158946853892.20572,
    "p90": 307513017942.04193
  },
  "hedging_value_mean_variance": null,
  "debug": {
    "pair_count": 320,
    "first_S_date": "2023-08-28",
    "last_S_date": "2025-06-06"
  }
}
```
