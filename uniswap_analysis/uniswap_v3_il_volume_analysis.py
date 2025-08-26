#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uniswap V3 — Small-Volume Pools: Daily IL via Implied v2 Depth (from Net Flow)

What this script does
---------------------
1) Queries the Uniswap v3 Subgraph (Subgraph Studio deployment) to fetch poolDayDatas
   for the last N days (default 30) for all Ethereum pools where daily volumeUSD is within
   the specified range (default: 0 to 10,000 USD). This selection is done **once up-front**
   to determine the pool set (as requested). We do not re-apply the daily volume filter later.

2) For those pools, fetches:
   - All poolDayDatas since the start date (prices/sqrtPrice, liquidity, tokens, etc.)
   - All Swaps since the start date (timestamp, amount0/amount1, pool)

3) For each pool-day, reconstructs an **implied v2 invariant k** from the **net swap flow**
   and **day-over-day price move**:
       Let P = price(token0 in token1) = (sqrtP)^2
       P0 = previous day's close, P1 = current day's close
       Δx = sum of swaps.amount0 for the day (pool perspective, signed; positive if pool gained token0)
       Then:
         Δx = sqrt(k)*(1/sqrt(P1) - 1/sqrt(P0))  =>  sqrt(k) = Δx / (1/sqrt(P1) - 1/sqrt(P0))
         k   = (sqrt(k))^2
       (We also compute sqrt(k) from Δy if available as a cross-check.)

   The **start-of-day notional** in token1 units is V_{0,1} = 2*sqrt(k*P0).
   The **IL fraction** is the standard v2 IL fraction for r = P1/P0:
       IL(r) = 1 - (2*sqrt(r)/(1+r)), clamped to [0, ∞).
   Per-day IL in token1 units is: IL_token1 = IL(r) * V_{0,1}.

   This matches the “implied depth from net in/out + price change” plan and
   automatically excludes LP mints/burns because we only aggregate **swaps**.

4) Valuation and ratios:
   - You can choose `--valuation token1` (no USD; no look-ahead) or `--valuation usd_current`
     (convert token1→USD using *current* token1.derivedETH * bundle.ethPriceUSD; introduces look-ahead).
   - Fees are approximated as volumeUSD * (feeTier/1e6). We also report IL/Fees.

Outputs
-------
- CSV file with per-pool summary:
    pool_id, tokens, feeTier, avg_daily_volumeUSD, avg_daily_IL_token1, avg_daily_IL_USD(optional),
    IL/Volume (if USD chosen), IL/Fees
- Histogram PNG of per-pool IL/Volume (or IL/Fees) distribution if applicable.
- Console prints with ratio stats and the top pools.

Caveats
-------
- We approximate v3 exposure by a v2-like notional derived from net flow and price change.
  When price traverses beyond most liquidity ranges, approximation error increases.
- USD valuation with `usd_current` uses **today’s** prices; this is a look-ahead convenience,
  left intentionally due to your constraint to avoid extra API requests.
- Aggregation: pools are selected by daily volume range up-front, then all days are included
  (no re-application of daily filter), exactly as requested.

Requirements
------------
- Python 3.9+
- requests, pandas, numpy, matplotlib

Usage
-----
python uniswap_v3_il_implied_k.py \
  --days 30 \
  --volume-min 0 \
  --volume-max 10000 \
  --endpoint https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV \
  --valuation usd_current \
  --out pools_il_volume.csv

Valuation options:
  - token1     : keep IL in token1 units (no USD; no look-ahead)
  - usd_current: convert IL_token1 to USD using current token1.derivedETH * bundle.ethPriceUSD (look-ahead)

"""

import argparse
import math
import os
import time
import sys
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# GraphQL helpers
# ---------------------------


class SubgraphClient:
    def __init__(self, retries: int = 3, delay: float = 1.0) -> None:
        self.retries = retries
        self.delay = delay
        self.headers = {"Content-Type": "application/json"}
        if k := os.getenv("THE_GRAPH_API_KEY"):
            self.headers["Authorization"] = f"Bearer {k}"

    def query(self, url: str, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"query": query, "variables": variables}
        err: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                r = requests.post(url, json=payload, headers=self.headers, timeout=60)
                r.raise_for_status()
                data = r.json()
                if "errors" in data:
                    raise RuntimeError(data["errors"])
                return data["data"]
            except Exception as exc:
                err = exc
                time.sleep(self.delay * 2**attempt)
        raise RuntimeError(err)


POOL_DAY_QUERY = """
query poolDayDatas($date_gte: Int!, $skip: Int!, $volume_lt: BigDecimal!, $volume_gte: BigDecimal!) {
  poolDayDatas(
    first: 1000
    skip: $skip
    orderBy: date
    orderDirection: asc
    where: { date_gte: $date_gte, volumeUSD_lt: $volume_lt, volumeUSD_gte: $volume_gte }
  ) {
    date
    volumeUSD
    sqrtPrice
    token0Price
    token1Price
    pool {
      id
      feeTier
      totalValueLockedUSD
      token0 { id symbol decimals derivedETH }
      token1 { id symbol decimals derivedETH }
    }
  }
}
"""

HISTORICAL_DAY_QUERY = """
query historicalData($date_gte: Int!, $skip: Int!, $pool_ids: [String!]!) {
  poolDayDatas(
    first: 1000
    skip: $skip
    orderBy: date
    orderDirection: asc
    where: { date_gte: $date_gte, pool_in: $pool_ids }
  ) {
    date
    volumeUSD
    sqrtPrice
    token0Price
    token1Price
    pool {
      id
      feeTier
      totalValueLockedUSD
      token0 { id symbol decimals derivedETH }
      token1 { id symbol decimals derivedETH }
    }
  }
}
"""

SWAPS_QUERY = """
query swaps($ts_gte: Int!, $skip: Int!, $pool_ids: [String!]!) {
  swaps(
    first: 1000
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
    where: { timestamp_gte: $ts_gte, pool_in: $pool_ids }
  ) {
    timestamp
    amount0
    amount1
    pool { id }
  }
}
"""

BUNDLE_QUERY = """
query bundle {
  bundle(id: "1") {
    ethPriceUSD
  }
}
"""

# ---------------------------
# Math helpers
# ---------------------------


def il_loss_fraction(r: float) -> float:
    """Impermanent loss fraction (positive; vs. HODL) for a Uniswap v2 AMM given r = P1/P0."""
    try:
        if r is None or r <= 0 or not math.isfinite(r):
            return 0.0
        root = math.sqrt(r)
        denom = 1.0 + r
        if denom == 0:
            return 0.0
        frac = 1.0 - (2.0 * root / denom)
        return max(0.0, frac)
    except Exception:
        return 0.0


def q96_to_sqrtp(q96: float) -> Optional[float]:
    """Convert Q64.96 sqrtPrice (as a float) to a floating sqrt(P)."""
    try:
        if q96 is None:
            return None
        return float(q96) / (2**96)
    except Exception:
        return None


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def day_bucket(ts: int) -> int:
    """UTC day bucket (Unix timestamp at 00:00:00)."""
    return (ts // 86400) * 86400


# ---------------------------
# Data fetching
# ---------------------------


def fetch_pools_with_volume_in_range(
    endpoint: str, start_ts: int, volume_min_usd: float, volume_max_usd: float
) -> List[str]:
    """Find pool IDs that have volume in the specified range within the time window (up-front selection)."""
    client = SubgraphClient()
    pool_ids = set()
    skip = 0

    print(
        f"Selecting pools with at least one day of volume in [{volume_min_usd:.2f}, {volume_max_usd:.2f})..."
    )

    while True:
        data = client.query(
            endpoint,
            POOL_DAY_QUERY,
            {
                "date_gte": int(start_ts),
                "skip": skip,
                "volume_lt": str(volume_max_usd),
                "volume_gte": str(volume_min_usd),
            },
        )
        chunk = data["poolDayDatas"]
        if not chunk:
            break
        for day_data in chunk:
            pool_ids.add(day_data["pool"]["id"])
        if len(chunk) < 1000:
            break
        skip += 1000

    print(f"Found {len(pool_ids)} unique pools.")
    return list(pool_ids)


def fetch_historical_days(
    endpoint: str, pool_ids: List[str], start_ts: int
) -> List[dict]:
    """Fetch all poolDayDatas for specific pools since start_ts."""
    if not pool_ids:
        return []
    client = SubgraphClient()
    results: List[dict] = []
    skip = 0

    print(f"Fetching poolDayDatas for {len(pool_ids)} pools...")
    while True:
        data = client.query(
            endpoint,
            HISTORICAL_DAY_QUERY,
            {
                "date_gte": int(start_ts),
                "skip": skip,
                "pool_ids": pool_ids,
            },
        )
        chunk = data["poolDayDatas"]
        if not chunk:
            break
        results.extend(chunk)
        if len(chunk) < 1000:
            break
        skip += 1000

    print(f"Fetched {len(results)} pool-day records.")
    return results


def fetch_swaps(endpoint: str, pool_ids: List[str], start_ts: int) -> List[dict]:
    """Fetch all swaps for the given pools since start_ts (paged)."""
    if not pool_ids:
        return []
    client = SubgraphClient()
    results: List[dict] = []
    skip = 0
    print(
        f"Fetching swaps since {time.strftime('%Y-%m-%d', time.gmtime(start_ts))} for {len(pool_ids)} pools..."
    )
    while True:
        data = client.query(
            endpoint,
            SWAPS_QUERY,
            {"ts_gte": int(start_ts), "skip": skip, "pool_ids": pool_ids},
        )
        chunk = data["swaps"]
        if not chunk:
            break
        results.extend(chunk)
        if len(chunk) < 1000:
            break
        skip += 1000
    print(f"Fetched {len(results)} swaps.")
    return results


def fetch_eth_usd(endpoint: str) -> Optional[float]:
    """Fetch current ETH/USD for optional USD valuation (look-ahead)."""
    try:
        client = SubgraphClient()
        data = client.query(endpoint, BUNDLE_QUERY, {})
        return float(data["bundle"]["ethPriceUSD"])
    except Exception:
        return None


# ---------------------------
# Processing & IL computation
# ---------------------------


def build_days_df(pool_days: List[dict]) -> pd.DataFrame:
    rows = []
    for d in pool_days:
        pool = d["pool"]
        rows.append(
            {
                "pool_id": pool["id"],
                "feeTier": int(pool["feeTier"]),
                "token0_symbol": pool["token0"]["symbol"],
                "token1_symbol": pool["token1"]["symbol"],
                "token0_decimals": int(pool["token0"]["decimals"]),
                "token1_decimals": int(pool["token1"]["decimals"]),
                "token0_derivedETH_current": safe_float(
                    pool["token0"].get("derivedETH")
                ),
                "token1_derivedETH_current": safe_float(
                    pool["token1"].get("derivedETH")
                ),
                "pool_tvlUSD_current": safe_float(pool.get("totalValueLockedUSD")),
                "date": int(d["date"]),  # UTC midnight bucket from subgraph
                "volumeUSD": safe_float(d["volumeUSD"]),
                "sqrtPriceQ96": safe_float(d["sqrtPrice"]),
                "token0Price": safe_float(
                    d["token0Price"]
                ),  # price of token0 in token1
                "token1Price": safe_float(d["token1Price"]),  # 1/token0Price
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values(["pool_id", "date"], inplace=True)
    # Convert sqrtPrice
    df["sqrtP"] = df["sqrtPriceQ96"].apply(q96_to_sqrtp)
    # Previous-day values
    df["sqrtP_prev"] = df.groupby("pool_id")["sqrtP"].shift(1)
    df["token0Price_prev"] = df.groupby("pool_id")["token0Price"].shift(1)
    # Prices
    df["P0"] = df["sqrtP_prev"] ** 2
    df["P1"] = df["sqrtP"] ** 2
    # Ratio
    df["r"] = df["P1"] / df["P0"]
    return df


def build_swaps_df(swaps: List[dict]) -> pd.DataFrame:
    rows = []
    for s in swaps:
        rows.append(
            {
                "pool_id": s["pool"]["id"],
                "timestamp": int(s["timestamp"]),
                "amount0": safe_float(s["amount0"]),  # signed, pool perspective
                "amount1": safe_float(s["amount1"]),  # signed, pool perspective
                "date": day_bucket(int(s["timestamp"])),  # align to UTC midnight bucket
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Aggregate by pool_id, date
    grouped = df.groupby(["pool_id", "date"], as_index=False).agg(
        net_amount0=("amount0", "sum"),
        net_amount1=("amount1", "sum"),
        swaps_count=("timestamp", "count"),
    )
    return grouped


def implied_sqrt_k_from_flows(
    P0: float, P1: float, dX: Optional[float], dY: Optional[float]
) -> Tuple[Optional[float], str]:
    """
    Infer sqrt(k) from Δx and/or Δy using:
      Δx = sqrt(k) * (1/sqrt(P1) - 1/sqrt(P0))
      Δy = sqrt(k) * (sqrt(P1) - sqrt(P0))
    Returns (sqrt_k, method), where method is 'dx', 'dy', or 'both' (if consistent).
    Preference: use both if they agree within tolerance; otherwise use the one with larger denominator magnitude.
    """
    if P0 is None or P1 is None or P0 <= 0 or P1 <= 0:
        return (None, "na")

    sp0 = math.sqrt(P0)
    sp1 = math.sqrt(P1)

    sqrt_k_dx = None
    sqrt_k_dy = None
    used = "na"

    # From Δx
    if dX is not None and math.isfinite(dX):
        denx = (1.0 / sp1) - (1.0 / sp0)
        if denx != 0 and math.isfinite(denx):
            sqrt_k_dx = dX / denx

    # From Δy
    if dY is not None and math.isfinite(dY):
        deny = sp1 - sp0
        if deny != 0 and math.isfinite(deny):
            sqrt_k_dy = dY / deny

    if (
        sqrt_k_dx is not None
        and sqrt_k_dy is not None
        and math.isfinite(sqrt_k_dx)
        and math.isfinite(sqrt_k_dy)
    ):
        # Check consistency within 5%
        if sqrt_k_dx == 0:
            rel_err = abs(sqrt_k_dy)
        else:
            rel_err = abs(sqrt_k_dx - sqrt_k_dy) / (abs(sqrt_k_dx) + 1e-12)
        if rel_err <= 0.05:
            return ((sqrt_k_dx + sqrt_k_dy) / 2.0, "both")
        # Otherwise, choose the one with larger |denominator| (numerically more stable)
        # Compare |1/sp1 - 1/sp0| vs |sp1 - sp0|
        if abs((1.0 / sp1) - (1.0 / sp0)) >= abs(sp1 - sp0):
            used = "dx"
            return (sqrt_k_dx, used)
        else:
            used = "dy"
            return (sqrt_k_dy, used)

    if sqrt_k_dx is not None and math.isfinite(sqrt_k_dx):
        return (sqrt_k_dx, "dx")
    if sqrt_k_dy is not None and math.isfinite(sqrt_k_dy):
        return (sqrt_k_dy, "dy")
    return (None, "na")


def compute_daily_il_implied_k(
    days_df: pd.DataFrame,
    swaps_df: pd.DataFrame,
    valuation: str,
    eth_usd_current: Optional[float],
) -> pd.DataFrame:
    """
    Merge days and swaps; compute implied sqrt(k), k, start-of-day notional, and IL per day.
    """
    if days_df.empty:
        return pd.DataFrame()

    # Merge net flows onto day rows
    df = days_df.merge(
        swaps_df,
        how="left",
        on=["pool_id", "date"],
        validate="m:1",
    )

    # Compute IL fraction from r
    df["il_frac"] = df["r"].apply(
        lambda v: il_loss_fraction(v) if v and math.isfinite(v) else 0.0
    )

    # Implied sqrt(k) and start-of-day notional in token1 units: V1 = 2*sqrt(k*P0)
    sqrt_ks = []
    methods = []
    V1s = []

    for row in df.itertuples(index=False):
        P0 = getattr(row, "P0")
        P1 = getattr(row, "P1")
        dX = getattr(row, "net_amount0", None)
        dY = getattr(row, "net_amount1", None)

        sqrt_k, method = implied_sqrt_k_from_flows(P0, P1, dX, dY)
        methods.append(method)
        if (
            sqrt_k is None
            or not math.isfinite(sqrt_k)
            or sqrt_k <= 0
            or P0 is None
            or P0 <= 0
        ):
            sqrt_ks.append(None)
            V1s.append(None)
        else:
            sqrt_ks.append(sqrt_k)
            V1s.append(2.0 * math.sqrt(sqrt_k * sqrt_k * P0))

    df["sqrt_k"] = sqrt_ks
    df["implied_method"] = methods
    df["notional_token1"] = V1s

    # IL in token1 units
    df["il_token1"] = df["il_frac"] * df["notional_token1"]

    # Optional USD valuation (look-ahead, no extra per-day requests)
    if valuation == "usd_current":
        # token1.derivedETH_current is from the pool entity embedded in each day row (current)
        token1_eth = df["token1_derivedETH_current"]
        if eth_usd_current and eth_usd_current > 0:
            df["token1_usd_now"] = token1_eth * eth_usd_current
            df["il_usd"] = df["il_token1"] * df["token1_usd_now"]
        else:
            df["token1_usd_now"] = np.nan
            df["il_usd"] = np.nan
    elif valuation == "token1":
        df["il_usd"] = np.nan
        df["token1_usd_now"] = np.nan
    else:
        raise ValueError("valuation must be one of: token1, usd_current")

    # Fees and ratios
    # feeTier is in hundredths of a bip (e.g., 500=0.05%, 3000=0.3%, 10000=1%)
    df["fees_usd_est"] = df["volumeUSD"] * (df["feeTier"] / 1e6)
    # Ratios: define both, but they will be NaN when units mismatch
    df["il_to_volume"] = df["il_usd"] / df["volumeUSD"]
    df["il_to_fees"] = df["il_usd"] / df["fees_usd_est"]

    # Keep only rows where we had a prior-day price and a valid sqrt_k-based notional
    df = df[
        (df["token0Price_prev"].notna())
        & (df["notional_token1"].notna())
        & (df["il_frac"].notna())
    ]

    return df


def aggregate_per_pool(daily_df: pd.DataFrame, valuation: str) -> pd.DataFrame:
    """
    Aggregate daily metrics to per-pool averages (no reapplication of daily volume filter).
    """
    if daily_df.empty:
        return pd.DataFrame()

    agg = {
        "volumeUSD": "mean",
        "il_token1": "mean",
        "fees_usd_est": "mean",
        "date": "count",
    }
    if valuation == "usd_current":
        agg["il_usd"] = "mean"
        agg["il_to_volume"] = "mean"
        agg["il_to_fees"] = "mean"

    grouped = daily_df.groupby(
        ["pool_id", "feeTier", "token0_symbol", "token1_symbol"], as_index=False
    ).agg(agg)
    grouped.rename(
        columns={
            "volumeUSD": "avg_daily_volume_usd",
            "il_token1": "avg_daily_il_token1",
            "il_usd": "avg_daily_il_usd",
            "fees_usd_est": "avg_daily_fees_usd",
            "date": "days",
            "il_to_volume": "avg_il_to_volume",
            "il_to_fees": "avg_il_to_fees",
        },
        inplace=True,
    )
    return grouped


# ---------------------------
# Main
# ---------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Uniswap v3: IL via implied v2 depth (from net swap flow) for pools in a daily volume range"
    )
    ap.add_argument(
        "--days", type=int, default=30, help="Lookback window in days (default: 30)"
    )
    ap.add_argument(
        "--volume-min",
        type=float,
        default=0.0,
        help="Minimum daily volumeUSD for pool selection (default: 0)",
    )
    ap.add_argument(
        "--volume-max",
        type=float,
        default=10000.0,
        help="Maximum daily volumeUSD for pool selection (default: 10,000)",
    )
    ap.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="Subgraph Studio GraphQL endpoint URL for Uniswap v3",
    )
    ap.add_argument(
        "--valuation",
        type=str,
        default="usd_current",
        choices=["token1", "usd_current"],
        help="Value IL in token1 units (no USD) or USD using current token1 USD price (look-ahead). Default: usd_current",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "pools_il_volume.csv"),
        help="Output CSV filename for per-pool summary",
    )
    args = ap.parse_args()

    # Start of window (UTC)
    now = int(time.time())
    start_ts = now - args.days * 86400

    print(
        f"Selecting pools since {time.strftime('%Y-%m-%d', time.gmtime(start_ts))} "
        f"with {args.volume_min:,.2f} <= daily volumeUSD < {args.volume_max:,.2f} ...",
        file=sys.stderr,
    )

    # 1) Pool selection by daily volume range (up-front)
    try:
        pool_ids = fetch_pools_with_volume_in_range(
            args.endpoint, start_ts, args.volume_min, args.volume_max
        )
    except Exception as e:
        print(f"ERROR fetching candidate pools: {e}", file=sys.stderr)
        sys.exit(1)

    if not pool_ids:
        print(
            "No pools matched the selection criteria. Nothing to do.", file=sys.stderr
        )
        sys.exit(0)

    # 2) Fetch all days + swaps for these pools
    try:
        pool_days = fetch_historical_days(args.endpoint, pool_ids, start_ts)
        swaps = fetch_swaps(args.endpoint, pool_ids, start_ts)
    except Exception as e:
        print(f"ERROR fetching history: {e}", file=sys.stderr)
        sys.exit(1)

    if not pool_days:
        print(
            "No poolDayDatas found for the selected pools. Nothing to do.",
            file=sys.stderr,
        )
        sys.exit(0)

    days_df = build_days_df(pool_days)
    swaps_df = build_swaps_df(swaps)

    # Optional: current ETH/USD for usd_current valuation
    eth_usd = None
    if args.valuation == "usd_current":
        eth_usd = fetch_eth_usd(args.endpoint)

    # 3) Compute daily IL via implied k
    daily_df = compute_daily_il_implied_k(days_df, swaps_df, args.valuation, eth_usd)

    if daily_df.empty:
        print(
            "No daily IL rows could be computed (missing prior-day price or insufficient swap flow).",
            file=sys.stderr,
        )
        sys.exit(0)

    # 4) Aggregate per pool
    summary_df = aggregate_per_pool(daily_df, args.valuation)
    if summary_df.empty:
        print("No per-pool aggregates could be computed.", file=sys.stderr)
        sys.exit(0)

    # 5) Save per-pool summary CSV
    if args.valuation == "usd_current":
        summary_cols = [
            "pool_id",
            "token0_symbol",
            "token1_symbol",
            "feeTier",
            "avg_daily_volume_usd",
            "avg_daily_fees_usd",
            "avg_daily_il_token1",
            "avg_daily_il_usd",
            "avg_il_to_volume",
            "avg_il_to_fees",
            "days",
        ]
    else:
        summary_cols = [
            "pool_id",
            "token0_symbol",
            "token1_symbol",
            "feeTier",
            "avg_daily_volume_usd",
            "avg_daily_fees_usd",
            "avg_daily_il_token1",
            "days",
        ]

    summary_df[summary_cols].to_csv(args.out, index=False)
    print("\nPer-pool summary saved to:", args.out)
    print(f"Pools included: {len(summary_df)}")

    # 6) Ratio stats and histogram (if USD valuation)
    if args.valuation == "usd_current":
        valid_ratios = (
            summary_df["avg_il_to_volume"].replace([np.inf, -np.inf], np.nan).dropna()
        )
        # Apply 99th percentile filter for consistent statistics
        if len(valid_ratios) > 0:
            q99 = np.percentile(valid_ratios, 99)
            filtered_ratios = valid_ratios[(valid_ratios > 0) & (valid_ratios <= q99)]
            if len(filtered_ratios) > 0:
                avg_ratio = filtered_ratios.mean()
                median_ratio = filtered_ratios.median()
                percentiles = np.percentile(filtered_ratios, [10, 25, 75, 90])
                p10, p25, p75, p90 = percentiles
            else:
                avg_ratio = median_ratio = p10 = p25 = p75 = p90 = float("nan")
        else:
            avg_ratio = median_ratio = p10 = p25 = p75 = p90 = float("nan")

        print("\nRatio Statistics (IL/Volume per pool, USD valuation):")
        print(f"  Average ratio: {avg_ratio:.6f}")
        print(f"  Median ratio: {median_ratio:.6f}")
        print(f"  10th percentile: {p10:.6f}")
        print(f"  25th percentile: {p25:.6f}")
        print(f"  75th percentile: {p75:.6f}")
        print(f"  90th percentile: {p90:.6f}")

        # Histogram of per-pool IL/Volume (USD)
        if len(valid_ratios) > 0:
            plt.figure(figsize=(12, 6))
            q99 = np.percentile(valid_ratios, 99)
            filtered = valid_ratios[(valid_ratios > 0) & (valid_ratios <= q99)]
            if len(filtered) > 0:
                log_bins = np.logspace(-4.5, 2, 30)
                counts, bin_edges = np.histogram(filtered, bins=log_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_widths = bin_edges[1:] - bin_edges[:-1]
                bars = plt.bar(
                    bin_centers,
                    counts,
                    width=bin_widths,
                    alpha=0.7,
                    edgecolor="black",
                    align="center",
                )
                for bar, left_edge, right_edge, count in zip(
                    bars, bin_edges[:-1], bin_edges[1:], counts
                ):
                    if count > 0:
                        left_str = f"{left_edge:.3g}"
                        right_str = f"{right_edge:.3g}"
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(counts) * 0.01,
                            f"{left_str}-{right_str}",
                            ha="center",
                            va="bottom",
                            rotation=45,
                            fontsize=8,
                        )
                plt.axvline(
                    avg_ratio,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {avg_ratio:.6f}",
                )
                plt.axvline(
                    median_ratio,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Median: {median_ratio:.6f}",
                )
                plt.xscale("log")
                plt.xlabel("Per-pool Average (IL USD / Volume USD) — log scale")
                plt.ylabel("Number of Pools")
                plt.title(
                    "Distribution of Per-Pool IL/Volume Ratios (USD valuation)\n(99th percentile truncated)"
                )
                plt.legend()
                plt.grid(True, alpha=0.3, which="both")
                plot_filename = args.out.replace(".csv", "_distribution.png")
                plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                print(f"\nRatio distribution plot saved to: {plot_filename}")
            else:
                print("No positive ratios found for visualization.")

    # 7) Top 10 pools by IL/Fees (if USD valuation) or by avg IL_token1 otherwise
    print("\nTop 10 pools:")
    if args.valuation == "usd_current" and "avg_il_to_fees" in summary_df.columns:
        top = summary_df.sort_values("avg_il_to_fees", ascending=False).head(10)
    else:
        top = summary_df.sort_values("avg_daily_il_token1", ascending=False).head(10)

    with pd.option_context("display.max_colwidth", 80):
        print(top[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
