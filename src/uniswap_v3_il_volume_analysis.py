#!/usr/bin/env python3
"""
Uniswap V3 — Small-Volume Pools: Daily IL vs Volume

What this script does
---------------------
1) Queries the Uniswap v3 Subgraph (Subgraph Studio deployment) to fetch poolDayDatas
   for the last N days (default 30) for all Ethereum pools where daily volumeUSD is within
   the specified range (default: 0 to 10,000 USD).

2) For each pool, computes an estimated daily impermanent loss (IL) using the Uniswap v2
   IL formula based on day-over-day price ratio r:
       IL_loss_fraction = 1 - (2 * sqrt(r) / (1 + r))
   where r = token0Price_today / token0Price_yesterday.

   To align with the instruction to "use the depth near the current tick to estimate IL
   as though it were a Uniswap v2 pool", we treat the active liquidity at the beginning of
   each day (approximated by the previous day's liquidity and sqrtPrice) as defining local
   "virtual reserves" X ~ L / sqrt(P) and Y ~ L * sqrt(P). The IL fraction itself does not
   require L, but to convert the fraction into USD IL per day, we scale by a USD notional.
   By default, we use the pool's totalValueLockedUSD (current) from the pool entity as a
   proxy notional. You can switch to an active-liquidity-derived notional with
   --notional-source virtual (see below).

3) For each pool, reports:
   - average daily Volume (USD)
   - average daily IL (USD, estimated)
   - ratio = (average daily IL) / (average daily Volume)

4) Also reports a global ratio computed as:
       Global Ratio = (sum over pools of average daily IL) / (sum over pools of average daily Volume)

Caveats
-------
- The Uniswap v3 subgraph provides daily aggregates. We use the day-over-day close price
  to approximate daily IL. This does not capture intra-day round-trips that would net
  IL back toward zero; it measures IL corresponding to daily net price displacement.
- The "depth near the current tick" concept is satisfied by using the active liquidity (L)
  and sqrtPrice to construct v2-like "virtual reserves". Since the v2 IL fraction depends
  only on the price ratio r, the main role of depth is to scale from fraction to USD.
  We default to pool.totalValueLockedUSD as a stable notional, but you can switch to an
  active-liquidity-derived notional with --notional-source virtual.
- Filtering by volumeUSD within the min/max range is done per day over the chosen lookback window.
  The per-pool averages and ratios then aggregate only those days within the volume range.

Requirements
------------
- Python 3.9+
- requests, pandas

Usage
-----
python uniswap_v3_il_volume_analysis.py \
  --days 30 \
  --volume-min 0 \
  --volume-max 10000 \
  --endpoint https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV
  --notional-source tvl \
  --out pools_il_volume.csv

--notional-source options:
  - tvl     : use pool.totalValueLockedUSD (default; robust)
  - virtual : use 2 * L * sqrtP * (token1 USD), where token1 USD is approximated via current
              token1.derivedETH * bundle.ethPriceUSD (falls back to pool TVL if unavailable)

Outputs
-------
- CSV file with per-pool summary (id, tokens, feeTier, avgDailyVolumeUSD, avgDailyILUSD, ratio)
- Prints a global ratio in the console.
"""

import argparse
import math
import os
import time
import sys
from typing import Any, List, Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# GraphQL helpers
# ---------------------------


class SubgraphClient:
    def __init__(self, retries: int = 3, delay: float = 1.0) -> None:
        self.retries, self.delay = retries, delay
        self.headers = {"Content-Type": "application/json"}
        if k := os.getenv("THE_GRAPH_API_KEY"):
            self.headers["Authorization"] = f"Bearer {k}"

    def query(self, url: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = {"query": query, "variables": variables}
        err: Exception | None = None
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
    liquidity
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

# For getting the ETH price in USD (bundle) if needed for "virtual" notionals.
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
    """
    Impermanent loss fraction (positive; vs. HODL) for a Uniswap v2 AMM
    given price ratio r = P_new / P_old.

    IL_loss_fraction = 1 - (2 * sqrt(r) / (1 + r))
    """
    try:
        if r <= 0 or not math.isfinite(r):
            return 0.0
        root = math.sqrt(r)
        denom = 1.0 + r
        if denom == 0:
            return 0.0
        frac = 1.0 - (2.0 * root / denom)
        # Numerical safety for r ~ 1
        return max(0.0, frac)
    except Exception:
        return 0.0


def q96_to_sqrtp(q96: float) -> float:
    """Convert Q64.96 sqrtPrice to a floating sqrt(P)."""
    return q96 / (2**96)


def virtual_notional_usd_from_L_P(
    L: float, sqrtP: float, token1_price_usd: Optional[float]
) -> Optional[float]:
    """
    For active liquidity L at price P, the local v2-like value (in token1 units) is:
      X*P + Y where X ~ L / sqrt(P) and Y ~ L * sqrt(P) => 2 * L * sqrt(P).
    Multiply by token1 USD price to obtain a USD notional tied to active depth.
    """
    if L is None or sqrtP is None or token1_price_usd is None:
        return None
    if L <= 0 or sqrtP <= 0 or token1_price_usd <= 0:
        return None
    return 2.0 * L * sqrtP * token1_price_usd


# ---------------------------
# Data fetching & processing
# ---------------------------


def fetch_pool_days(
    endpoint: str, start_ts: int, volume_min_usd: float, volume_max_usd: float
) -> List[dict]:
    """Pull all poolDayDatas since start_ts where volume_min_usd <= volumeUSD < volume_max_usd, with pagination."""
    client = SubgraphClient()
    results: List[dict] = []
    skip = 0
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
        results.extend(chunk)
        if len(chunk) < 1000:
            break
        skip += 1000
    return results


def fetch_eth_usd(endpoint: str) -> Optional[float]:
    try:
        client = SubgraphClient()
        data = client.query(endpoint, BUNDLE_QUERY, {})
        return float(data["bundle"]["ethPriceUSD"])
    except Exception:
        return None


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def compute_pool_metrics(
    pool_days: List[dict], notional_source: str, eth_usd: Optional[float]
):
    """
    Converts raw poolDay rows into per-pool daily metrics and then per-pool aggregates.
    Returns (daily_df, summary_df).
    """
    rows = []
    for d in pool_days:
        pool = d["pool"]
        row = {
            "pool_id": pool["id"],
            "feeTier": int(pool["feeTier"]),
            "token0_symbol": pool["token0"]["symbol"],
            "token1_symbol": pool["token1"]["symbol"],
            "token0_decimals": int(pool["token0"]["decimals"]),
            "token1_decimals": int(pool["token1"]["decimals"]),
            "token0_derivedETH": safe_float(pool["token0"].get("derivedETH")),
            "token1_derivedETH": safe_float(pool["token1"].get("derivedETH")),
            "pool_tvlUSD": safe_float(pool.get("totalValueLockedUSD")),
            "date": int(d["date"]),
            "volumeUSD": safe_float(d["volumeUSD"]),
            "sqrtPriceQ96": safe_float(d["sqrtPrice"]),
            "liquidity": safe_float(d["liquidity"]),
            "token0Price": safe_float(d["token0Price"]),
            "token1Price": safe_float(d["token1Price"]),
        }
        row["sqrtP"] = (
            q96_to_sqrtp(row["sqrtPriceQ96"]) if row["sqrtPriceQ96"] else None
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    df.sort_values(["pool_id", "date"], inplace=True)

    # Day-over-day price ratio r using token0Price
    df["token0Price_prev"] = df.groupby("pool_id")["token0Price"].shift(1)
    df["r"] = df["token0Price"] / df["token0Price_prev"]

    # IL loss fraction
    df["il_frac"] = df["r"].apply(
        lambda v: il_loss_fraction(v) if v and math.isfinite(v) else 0.0
    )

    # Notional for scaling IL fraction -> USD
    if notional_source == "tvl":
        df["notionalUSD"] = df["pool_tvlUSD"]
    elif notional_source == "virtual":
        # Use previous day's liquidity and sqrtP together with token1 USD price
        if eth_usd:
            df["token1USD_est"] = df["token1_derivedETH"].apply(
                lambda x: (
                    (x * eth_usd) if (x is not None and math.isfinite(x)) else None
                )
            )
        else:
            df["token1USD_est"] = None

        df["L_prev"] = df.groupby("pool_id")["liquidity"].shift(1)
        df["sqrtP_prev"] = df.groupby("pool_id")["sqrtP"].shift(1)

        def row_virtual_notional(row):
            return virtual_notional_usd_from_L_P(
                L=row.get("L_prev") or 0.0,
                sqrtP=row.get("sqrtP_prev") or 0.0,
                token1_price_usd=row.get("token1USD_est"),
            )

        df["notionalUSD"] = df.apply(row_virtual_notional, axis=1)
        # Fallback to TVL if virtual notional is missing/unavailable
        df.loc[df["notionalUSD"].isna(), "notionalUSD"] = df.loc[
            df["notionalUSD"].isna(), "pool_tvlUSD"
        ]
    else:
        raise ValueError("notional_source must be one of: tvl, virtual")

    # Daily IL in USD (estimated)
    df["il_usd"] = df["il_frac"] * df["notionalUSD"]
    # Filter to rows where a prior price exists and volume/notional are present
    df = df[
        (df["token0Price_prev"].notna())
        & (df["volumeUSD"].notna())
        & (df["notionalUSD"].notna())
    ]

    grouped = df.groupby(
        ["pool_id", "feeTier", "token0_symbol", "token1_symbol"], as_index=False
    ).agg(
        avg_daily_volume_usd=("volumeUSD", "mean"),
        avg_daily_il_usd=("il_usd", "mean"),
        days=("date", "count"),
    )
    grouped["ratio_il_to_volume"] = grouped.apply(
        lambda r: (
            (r["avg_daily_il_usd"] / r["avg_daily_volume_usd"])
            if r["avg_daily_volume_usd"]
            else float("nan")
        ),
        axis=1,
    )

    return df, grouped


def main():
    ap = argparse.ArgumentParser(
        description="Uniswap v3: IL vs Volume for pools within specified daily volume range"
    )
    ap.add_argument(
        "--days", type=int, default=30, help="Lookback window in days (default: 30)"
    )
    ap.add_argument(
        "--volume-min",
        type=float,
        default=0.0,
        help="Minimum daily volumeUSD threshold (default: 0)",
    )
    ap.add_argument(
        "--volume-max",
        type=float,
        default=10000.0,
        help="Maximum daily volumeUSD threshold (default: 10,000)",
    )
    ap.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="Subgraph Studio GraphQL endpoint URL for Uniswap v3",
    )
    ap.add_argument(
        "--notional-source",
        type=str,
        default="tvl",
        choices=["tvl", "virtual"],
        help="How to scale IL fraction to USD (default: tvl)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="pools_il_volume.csv",
        help="Output CSV filename for per-pool summary",
    )
    args = ap.parse_args()

    # Start of window (UTC)
    now = int(time.time())
    start_ts = now - args.days * 86400

    print(
        f"Fetching poolDayDatas since {time.strftime('%Y-%m-%d', time.gmtime(start_ts))} "
        f"with {args.volume_min:,.2f} <= volumeUSD < {args.volume_max:,.2f} ...",
        file=sys.stderr,
    )


    try:
        pool_days = fetch_pool_days(
            args.endpoint, start_ts, args.volume_min, args.volume_max
        )
    except Exception as e:
        print(f"ERROR fetching pool day data: {e}", file=sys.stderr)
        sys.exit(1)

    if not pool_days:
        print("No poolDayDatas matched the filters. Nothing to do.", file=sys.stderr)
        sys.exit(0)

    eth_usd = None
    if args.notional_source == "virtual":
        eth_usd = fetch_eth_usd(args.endpoint)

    daily_df, summary_df = compute_pool_metrics(
        pool_days, args.notional_source, eth_usd
    )

    if summary_df.empty:
        print(
            "No per-pool summary could be computed (possibly missing prior-day prices).",
            file=sys.stderr,
        )
        sys.exit(0)

    # Calculate statistics for individual pool ratios
    valid_ratios = summary_df["ratio_il_to_volume"].dropna()
    if len(valid_ratios) > 0:
        avg_ratio = valid_ratios.mean()
        median_ratio = valid_ratios.median()
        percentiles = np.percentile(valid_ratios, [10, 25, 75, 90])
        p10, p25, p75, p90 = percentiles
    else:
        avg_ratio = median_ratio = p10 = p25 = p75 = p90 = float("nan")

    # Keep the old calculation for comparison
    total_avg_il = summary_df["avg_daily_il_usd"].sum(skipna=True)
    total_avg_volume = summary_df["avg_daily_volume_usd"].sum(skipna=True)
    global_ratio_old = (
        (total_avg_il / total_avg_volume) if total_avg_volume else float("nan")
    )

    # Save per-pool summary
    summary_cols = [
        "pool_id",
        "token0_symbol",
        "token1_symbol",
        "feeTier",
        "avg_daily_volume_usd",
        "avg_daily_il_usd",
        "ratio_il_to_volume",
        "days",
    ]
    summary_df[summary_cols].to_csv(args.out, index=False)

    print("\nPer-pool summary saved to:", args.out)
    print(f"Pools included: {len(summary_df)}")
    print(f"Pools with valid ratios: {len(valid_ratios)}")
    print("\nRatio Statistics (IL/Volume per pool):")
    print(f"  Average ratio: {avg_ratio:.6f}")
    print(f"  Median ratio: {median_ratio:.6f}")
    print(f"  10th percentile: {p10:.6f}")
    print(f"  25th percentile: {p25:.6f}")
    print(f"  75th percentile: {p75:.6f}")
    print(f"  90th percentile: {p90:.6f}")
    print(
        f"\nFor comparison, old calculation (sum(IL)/sum(Volume)): {global_ratio_old:.6f}"
    )

    # Create histogram of ratio distribution
    if len(valid_ratios) > 0:
        plt.figure(figsize=(12, 6))

        # Remove extreme outliers and zero/negative values for log scale
        q99 = np.percentile(valid_ratios, 99)
        filtered_ratios = valid_ratios[(valid_ratios > 0) & (valid_ratios <= q99)]
        
        if len(filtered_ratios) == 0:
            print("No positive ratios found for visualization")
        else:
            # Create histogram with log bins - equal log width bars
            log_bins = np.logspace(np.log10(filtered_ratios.min()), np.log10(filtered_ratios.max()), 30)
            counts, bin_edges = np.histogram(filtered_ratios, bins=log_bins)
            
            # Calculate bin centers and widths in log space
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            
            # Create bar chart with proper log-scale widths
            bars = plt.bar(bin_centers, counts, width=bin_widths, alpha=0.7, edgecolor='black', align='center')
            
            # Add labels on top of bars showing the ratio range for each bin
            for i, (bar, left_edge, right_edge, count) in enumerate(zip(bars, bin_edges[:-1], bin_edges[1:], counts)):
                if count > 0:  # Only label bars with data
                    # Format the range nicely
                    if left_edge < 0.001:
                        left_str = f"{left_edge:.1e}"
                    else:
                        left_str = f"{left_edge:.3f}"
                    
                    if right_edge < 0.001:
                        right_str = f"{right_edge:.1e}"
                    else:
                        right_str = f"{right_edge:.3f}"
                    
                    # Place label on top of bar
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                            f"{left_str}-{right_str}", 
                            ha='center', va='bottom', rotation=45, fontsize=8)
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

        plt.xscale('log')
        plt.xlabel("IL/Volume Ratio (log scale)")
        plt.ylabel("Number of Pools")
        plt.title(
            "Distribution of IL/Volume Ratios Across Pools (Log Scale)\n(99th percentile and above excluded for clarity)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')

        # Save the plot
        plot_filename = args.out.replace(".csv", "_distribution.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"\nRatio distribution plot saved to: {plot_filename}")

    print("\nTop 10 pools by ratio (descending):")
    top10 = summary_df.sort_values("ratio_il_to_volume", ascending=False).head(10)
    with pd.option_context("display.max_colwidth", 80):
        print(top10[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
