# Hedging Standard Deviation — aave_yield / usdc

**Window (UTC):** 2025-08-10T00:00:00+00:00 → 2025-09-05T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Exposure to metric (window) | $39,472.42 |
| StdDev of hedging P&L (p50) | $3,161,911 |
| StdDev of hedging P&L (p80) | $10,614,547 |
| StdDev of hedging P&L (p90) | $13,422,213 |

## Inputs & Diagnostics
- S→R pairs used: `382` (from `2024-07-27` to `2025-08-12`)
- Metric % Std (period): `53.23%` (std=`215.88`, median level=`405.53`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "aave_yield",
  "profile": "usdc",
  "exposure_to_metric": 39472.42,
  "delta_abs_percentiles": {
    "p50": 80.1,
    "p80": 268.91,
    "p90": 340.04
  },
  "hedging_std_component": {
    "p50": 3161910.66,
    "p80": 10614547.16,
    "p90": 13422212.76
  },
  "debug": {
    "pair_count": 382,
    "first_S_date": "2024-07-27",
    "last_S_date": "2025-08-12",
    "metric_change_std": 215.88,
    "metric_level_median": 405.53
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/aave_yield__usdc__hedge_std__2025-08-10__to__2025-09-05.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/aave_yield__usdc__hedge_std__2025-08-10__to__2025-09-05.md"
  },
  "window": {
    "start": "2025-08-10T00:00:00+00:00",
    "end": "2025-09-05T00:00:00+00:00",
    "period_days": 26
  }
}
```
