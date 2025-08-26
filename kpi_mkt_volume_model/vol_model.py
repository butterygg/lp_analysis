#!/usr/bin/env python3
"""
Market Volume Forecaster (30D or any period)
--------------------------------------------
Forecasts notional trading volume over a chosen period as the minimum of:
  1) Demand capacity (funnel-bound)
  2) Liquidity capacity (depth × turnover × days)
  3) Subsidy capacity (rebate-budget bound)

Usage:
  - Edit the CONFIG section below and run:
      python forecast_volume.py
  - Set REBATE_RATE_BPS=0 or REBATE_BUDGET_USD=0 to disable subsidy cap.

All math is transparent and printed at the end.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import math
import json

# ==============================
# ========== CONFIG ============
# ==============================

# Identify the market and period
MARKET_NAME: str = "Base TVL Market"
PERIOD_DAYS: int = 30  # set to your market duration (e.g., 7, 30, 90)

# --- Funnel (Demand) inputs ---
INVITED_USERS: int = 2000  # N_invited
VISIT_RATE: float = 0.45  # fraction who visit (0..1)
TRADE_CONV_RATE: float = 0.25  # fraction of visitors who trade (0..1)
TRADES_PER_TRADER: float = 5.0  # average # of trades per trading user in the period
AVG_TICKET_USD: float = 300.0  # average notional per trade (USD)

# --- Liquidity inputs ---
DEPTH_1PCT_USD: float = (
    25_000.0  # effective near-mid depth that moves price ~1% (both sides)
)
DAILY_TURNOVER_MULTIPLE: float = 4.0  # how many times that depth turns over per day

# --- Subsidy/Rebate inputs ---
REBATE_BUDGET_USD: float = 1_700.0  # total rebate $ for the period
REBATE_RATE_BPS: float = 20.0  # taker rebate in basis points (e.g., 20 bps = 0.20%)
REBATE_UTILIZATION: float = 0.70  # fraction of budget that actually gets used (0..1)

# --- Output options ---
PRINT_JSON_SUMMARY: bool = True
SHOW_DAILY_BREAKDOWN: bool = False  # if True, prints a naive daily breakdown
DAILY_PROFILE: str = "uniform"  # "uniform" | "frontloaded" | "backloaded" | "custom"
CUSTOM_DAILY_WEIGHTS: List[float] = (
    []
)  # if DAILY_PROFILE == "custom", provide len==PERIOD_DAYS non-negative weights

# ==============================
# ===== END OF CONFIG ==========
# ==============================


def _assert_prob(name: str, x: float) -> None:
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"{name} must be in [0,1], got {x}")


def _assert_pos(name: str, x: float) -> None:
    if x <= 0:
        raise ValueError(f"{name} must be > 0, got {x}")


def _fmt_usd(x: float) -> str:
    if math.isinf(x):
        return "∞"
    return f"${x:,.0f}"


@dataclass
class Funnel:
    invited: int
    visit_rate: float
    trade_conv_rate: float
    trades_per_trader: float
    avg_ticket_usd: float

    def validate(self) -> None:
        _assert_pos(
            "invited", self.invited if self.invited > 0 else 1
        )  # allow invited=0 (yields 0 cap)
        _assert_prob("visit_rate", self.visit_rate)
        _assert_prob("trade_conv_rate", self.trade_conv_rate)
        _assert_pos("trades_per_trader", max(self.trades_per_trader, 1e-12))
        _assert_pos("avg_ticket_usd", max(self.avg_ticket_usd, 1e-12))

    def demand_cap(self) -> float:
        """V_demand = N_invited * r_visit * r_trade * trades_per_trader * avg_ticket"""
        traders = self.invited * self.visit_rate * self.trade_conv_rate
        return traders * self.trades_per_trader * self.avg_ticket_usd


@dataclass
class Liquidity:
    depth_1pct_usd: float
    daily_turnover_multiple: float
    days: int

    def validate(self) -> None:
        _assert_pos("depth_1pct_usd", max(self.depth_1pct_usd, 1e-12))
        _assert_pos("daily_turnover_multiple", max(self.daily_turnover_multiple, 1e-12))
        _assert_pos("days", max(self.days, 1))

    def liquidity_cap(self) -> float:
        """V_liquidity = depth_1pct_usd * daily_turnover_multiple * days"""
        return self.depth_1pct_usd * self.daily_turnover_multiple * self.days


@dataclass
class Subsidy:
    budget_usd: float
    rebate_rate_bps: float
    utilization: float

    def validate(self) -> None:
        # budget can be zero (meaning no subsidy cap)
        _assert_prob("utilization", self.utilization)

    def subsidy_cap(self) -> float:
        """
        V_subsidy = (budget * utilization) / rebate_rate
        If rebate rate <= 0 or budget == 0 or utilization == 0 => no binding cap (infinite).
        """
        if self.budget_usd <= 0 or self.utilization <= 0 or self.rebate_rate_bps <= 0:
            return float("inf")
        rebate_rate = self.rebate_rate_bps / 10_000.0  # bps -> decimal
        # guard against tiny bps leading to huge numbers; still valid mathematically
        return (self.budget_usd * self.utilization) / rebate_rate


def forecast_volume(funnel: Funnel, liq: Liquidity, sub: Subsidy) -> Dict[str, Any]:
    funnel.validate()
    liq.validate()
    sub.validate()

    vd = funnel.demand_cap()
    vl = liq.liquidity_cap()
    vs = sub.subsidy_cap()

    caps = [("Demand", vd), ("Liquidity", vl), ("Subsidy", vs)]
    binding, v_forecast = min(caps, key=lambda kv: kv[1])

    return {
        "V_demand": vd,
        "V_liquidity": vl,
        "V_subsidy": vs,
        "V_forecast": v_forecast,
        "binding_constraint": binding,
    }


def lever_sensitivities(
    funnel: Funnel, liq: Liquidity, sub: Subsidy, binding: str
) -> Dict[str, float]:
    """
    First-order partial derivatives of the *binding* capacity w.r.t. each lever.
    These are local/linear and only apply while the same constraint remains binding.
    """
    sens: Dict[str, float] = {}
    if binding == "Demand":
        sens["dV/d_invited"] = (
            funnel.visit_rate
            * funnel.trade_conv_rate
            * funnel.trades_per_trader
            * funnel.avg_ticket_usd
        )
        sens["dV/d_visit_rate"] = (
            funnel.invited
            * funnel.trade_conv_rate
            * funnel.trades_per_trader
            * funnel.avg_ticket_usd
        )
        sens["dV/d_trade_conv_rate"] = (
            funnel.invited
            * funnel.visit_rate
            * funnel.trades_per_trader
            * funnel.avg_ticket_usd
        )
        sens["dV/d_trades_per_trader"] = (
            funnel.invited
            * funnel.visit_rate
            * funnel.trade_conv_rate
            * funnel.avg_ticket_usd
        )
        sens["dV/d_avg_ticket_usd"] = (
            funnel.invited
            * funnel.visit_rate
            * funnel.trade_conv_rate
            * funnel.trades_per_trader
        )
    elif binding == "Liquidity":
        sens["dV/d_depth_1pct_usd"] = liq.daily_turnover_multiple * liq.days
        sens["dV/d_daily_turnover_multiple"] = liq.depth_1pct_usd * liq.days
        sens["dV/d_days"] = liq.depth_1pct_usd * liq.daily_turnover_multiple
    elif binding == "Subsidy":
        if sub.rebate_rate_bps > 0 and sub.utilization > 0:
            r = sub.rebate_rate_bps / 10_000.0
            sens["dV/d_budget_usd"] = sub.utilization / r
            sens["dV/d_utilization"] = sub.budget_usd / r
            sens["dV/d_rebate_rate_bps"] = -(sub.budget_usd * sub.utilization) / (
                (r**2) * 10_000.0
            )
        else:
            # No subsidy cap => no meaningful sensitivities here
            sens["note"] = 0.0
    return sens


def _daily_weights(days: int, profile: str, custom: List[float]) -> List[float]:
    if days <= 0:
        raise ValueError("days must be positive")
    if profile == "uniform":
        return [1.0 / days] * days
    elif profile == "frontloaded":
        # Quadratic decay to back
        raw = [((days - i) / days) ** 2 for i in range(days)]
    elif profile == "backloaded":
        # Quadratic rise to end
        raw = [((i + 1) / days) ** 2 for i in range(days)]
    elif profile == "custom":
        if len(custom) != days:
            raise ValueError(f"custom weights must have length {days}")
        raw = [max(0.0, w) for w in custom]
    else:
        raise ValueError(
            "DAILY_PROFILE must be one of: uniform | frontloaded | backloaded | custom"
        )

    s = sum(raw)
    if s <= 0:
        raise ValueError("Daily weights sum to zero")
    return [w / s for w in raw]


def daily_breakdown(
    vd: float, vl: float, vs: float, days: int, profile: str, custom: List[float]
) -> List[Dict[str, Any]]:
    """
    Simple daily allocation of each capacity using weights.
    Note: This is a naive split (not a dynamic fill process).
    """
    weights = _daily_weights(days, profile, custom)
    out = []
    for d in range(days):
        vdd = vd * weights[d]
        vll = vl / days  # liquidity cap assumed even per day
        if math.isinf(vs):
            vss = float("inf")
        else:
            vss = vs * weights[d] if profile != "uniform" else (vs / days)
        day_forecast = min(vdd, vll, vss)
        if day_forecast == vdd:
            bind = "Demand"
        elif day_forecast == vll:
            bind = "Liquidity"
        else:
            bind = "Subsidy"
        out.append(
            {
                "day": d + 1,
                "Vd": vdd,
                "Vl": vll,
                "Vs": vss,
                "V_day": day_forecast,
                "binding": bind,
            }
        )
    return out


def main() -> None:
    # Build inputs
    funnel = Funnel(
        invited=INVITED_USERS,
        visit_rate=VISIT_RATE,
        trade_conv_rate=TRADE_CONV_RATE,
        trades_per_trader=TRADES_PER_TRADER,
        avg_ticket_usd=AVG_TICKET_USD,
    )
    liq = Liquidity(
        depth_1pct_usd=DEPTH_1PCT_USD,
        daily_turnover_multiple=DAILY_TURNOVER_MULTIPLE,
        days=PERIOD_DAYS,
    )
    sub = Subsidy(
        budget_usd=REBATE_BUDGET_USD,
        rebate_rate_bps=REBATE_RATE_BPS,
        utilization=REBATE_UTILIZATION,
    )

    # Forecast
    result = forecast_volume(funnel, liq, sub)
    vd, vl, vs = result["V_demand"], result["V_liquidity"], result["V_subsidy"]
    v_forecast, binding = result["V_forecast"], result["binding_constraint"]

    # Print summary
    print("=" * 70)
    print(f"Market: {MARKET_NAME}")
    print(f"Period: {PERIOD_DAYS} days")
    print("-" * 70)
    print("Inputs:")
    print(
        f"  Demand: invited={INVITED_USERS}, visit_rate={VISIT_RATE:.2%}, "
        f"trade_conv_rate={TRADE_CONV_RATE:.2%}, trades_per_trader={TRADES_PER_TRADER}, "
        f"avg_ticket={_fmt_usd(AVG_TICKET_USD)}"
    )
    print(
        f"  Liquidity: depth_1pct={_fmt_usd(DEPTH_1PCT_USD)}, "
        f"daily_turnover_multiple={DAILY_TURNOVER_MULTIPLE}x"
    )
    print(
        f"  Subsidy: budget={_fmt_usd(REBATE_BUDGET_USD)}, "
        f"rebate_rate={REBATE_RATE_BPS:.2f} bps, utilization={REBATE_UTILIZATION:.0%}"
    )
    print("-" * 70)
    print("Capacities:")
    print(f"  V_demand   = {_fmt_usd(vd)}")
    print(f"  V_liquidity= {_fmt_usd(vl)}")
    print(f"  V_subsidy  = {_fmt_usd(vs)}")
    print("-" * 70)
    print(f"FORECAST (binding = {binding}): {_fmt_usd(v_forecast)} over {PERIOD_DAYS}d")
    print("=" * 70)

    # Sensitivities
    sens = lever_sensitivities(funnel, liq, sub, binding)
    if sens:
        print("\nLocal sensitivities (valid while the same constraint stays binding):")
        for k, v in sens.items():
            if k.startswith("note"):
                print("  (No subsidy cap active; sensitivities not applicable.)")
            else:
                print(f"  {k:>22s} ≈ {v:,.4f}")

    # Optional JSON
    if PRINT_JSON_SUMMARY:
        package = {
            "market": MARKET_NAME,
            "period_days": PERIOD_DAYS,
            "inputs": {
                "funnel": asdict(funnel),
                "liquidity": asdict(liq),
                "subsidy": asdict(sub),
            },
            "results": result,
        }
        print("\nJSON summary:")
        print(json.dumps(package, indent=2))

    # Optional daily breakdown
    if SHOW_DAILY_BREAKDOWN:
        rows = daily_breakdown(
            vd, vl, vs, PERIOD_DAYS, DAILY_PROFILE, CUSTOM_DAILY_WEIGHTS
        )
        print("\nDaily breakdown (naive allocation):")
        header = (
            f"{'Day':>3} | {'Vd':>12} | {'Vl':>12} | {'Vs':>12} | {'V_day':>12} | bind"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['day']:>3} | {_fmt_usd(r['Vd']):>12} | {_fmt_usd(r['Vl']):>12} | "
                f"{_fmt_usd(r['Vs']):>12} | {_fmt_usd(r['V_day']):>12} | {r['binding']}"
            )


if __name__ == "__main__":
    main()
