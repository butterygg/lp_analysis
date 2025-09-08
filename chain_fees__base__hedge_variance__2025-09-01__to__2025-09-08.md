# Hedging Variance Report

**KPI:** chain_fees  
**Profile:** base  
**Window (UTC):** 2025-09-01 → 2025-09-08  

## Inputs
- Exposure to metric: `0.019165`
- Fraction of portfolio exposed: `1.000000`
- Delta^2 over period (capped contemporaneously): p50=`3,702,965,904.000000`, p80=`24,223,038,280.600010`, p90=`45,305,589,659.200035`

## Hedging Variance Component
- p50: `1,360,138.643408`  
- p80: `8,897,378.825606`  
- p90: `16,641,223.509859`

## Diagnostics
- S->R pairs used: `363`  (from `2024-09-02` to `2025-08-30`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "base",
  "exposure_to_metric": 0.019165349048919554,
  "delta_squared_percentiles": {
    "p50": 3702965904.0,
    "p80": 24223038280.60001,
    "p90": 45305589659.200035
  },
  "hedging_variance_component": {
    "p50": 1360138.643407751,
    "p80": 8897378.82560566,
    "p90": 16641223.509859404
  },
  "hedging_value_mean_variance": null,
  "debug": {
    "pair_count": 363,
    "first_S_date": "2024-09-02",
    "last_S_date": "2025-08-30"
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/chain_fees__base__hedge_variance__2025-09-01__to__2025-09-08.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/chain_fees__base__hedge_variance__2025-09-01__to__2025-09-08.md"
  },
  "window": {
    "start": "2025-09-01",
    "end": "2025-09-08",
    "period_days": 7
  }
}
```
