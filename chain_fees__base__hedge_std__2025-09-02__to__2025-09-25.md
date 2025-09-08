# Hedging Standard Deviation — chain_fees / base

**Window (UTC):** 2025-09-02T00:00:00+00:00 → 2025-09-25T00:00:00+00:00  

## Summary

| Metric | Value |
|---|---:|
| Average total exposure (metric × exposure/unit) | $5,095,655.04 |
| StdDev of hedging P&L (p50) | $2,229,068 |
| StdDev of hedging P&L (p80) | $5,053,220 |
| StdDev of hedging P&L (p90) | $7,643,668 |

### Metric % Std (period)

| Metric | Value (%) |
|---|---:|
| Std/Median (over window) | 109.17% |

## Inputs & Diagnostics
- S→R pairs used: `347` (from `2024-09-02` to `2025-08-14`)
- Metric % Std (period): `109.17%` (std=`189,285.43`, median level=`173,392.00`)

## Raw Output (for programmatic use)
```json
{
  "kpi": "chain_fees",
  "profile": "base",
  "exposure_to_metric": 23.0,
  "average_total_exposure": 5095655.04,
  "delta_abs_percentiles": {
    "p50": 96916.0,
    "p80": 219705.2,
    "p90": 332333.4
  },
  "metric_percent_std_pct": 109.17,
  "hedging_std_component": {
    "p50": 2229068.0,
    "p80": 5053219.6,
    "p90": 7643668.2
  },
  "debug": {
    "pair_count": 347,
    "first_S_date": "2024-09-02",
    "last_S_date": "2025-08-14",
    "metric_change_std": 189285.43,
    "metric_level_median": 173392.0,
    "metric_level_mean": 221550.22
  },
  "report": {
    "path": "/home/pimania/dev/butter/lp_analysis/chain_fees__base__hedge_std__2025-09-02__to__2025-09-25.md",
    "url": "https://github.com/butterygg/lp_analysis/blob/master/chain_fees__base__hedge_std__2025-09-02__to__2025-09-25.md"
  },
  "window": {
    "start": "2025-09-02T00:00:00+00:00",
    "end": "2025-09-25T00:00:00+00:00",
    "period_days": 23
  }
}
```
