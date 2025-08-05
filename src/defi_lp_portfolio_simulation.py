#!/usr/bin/env python3
"""
DeFi LP Portfolio Simulation Script

This script:
1. Fetches top N DeFi protocols from DeFiLlama
2. Calculates historical TVL share of each protocol over 24 months
3. Simulates UP/DOWN token liquidity pool performance
4. Graphs portfolio performance with opportunity cost adjustment
"""

from __future__ import annotations

import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Configuration
TOP_N_PROTOCOLS = 10
ANALYSIS_MONTHS = 24
SIMULATION_PERIOD_DAYS = 30  # One month simulation periods
PERIOD_SPACING_DAYS = 1  # Number of days between the start of each period
ANNUAL_INTEREST_RATE = 0.25  # 25% annualized
MONTHLY_INTEREST_RATE = (1 + ANNUAL_INTEREST_RATE) ** (1 / 12) - 1  # ~2% per month

# Liquidity withdrawal configuration
WITHDRAWAL_ENABLED = True  # Enable partial liquidity withdrawal
WITHDRAWAL_TIMING_PCT = 0.25  # When to withdraw (0.5 = halfway through period)
WITHDRAWAL_AMOUNT_PCT = 0.7  # How much to withdraw (0.5 = 50% of liquidity)


def fetch_top_protocols(n: int = TOP_N_PROTOCOLS) -> List[Dict]:
    """Fetch top N DeFi protocols by TVL from DeFiLlama."""
    print(f"Fetching top {n} DeFi protocols...")

    url = "https://api.llama.fi/protocols"
    response = requests.get(url, timeout=30)

    if not response.ok:
        raise RuntimeError(f"Failed to fetch protocols: {response.status_code}")

    protocols = response.json()

    # Filter for DeFi protocols and sort by TVL
    defi_protocols = [p for p in protocols]
    defi_protocols.sort(key=lambda x: x.get("tvl", 0), reverse=True)

    if len(defi_protocols) < n:
        return defi_protocols

    top_protocols = defi_protocols[:n]
    print(f"Selected protocols: {[p['name'] for p in top_protocols]}")

    return top_protocols


def fetch_protocol_history(slug: str) -> Dict:
    """Fetch historical TVL data for a protocol."""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{slug}_history.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    url = f"https://api.llama.fi/protocol/{slug}"
    response = requests.get(url, timeout=30)

    if not response.ok:
        print(f"Failed to fetch {slug}: {response.status_code}")
        return {}

    data = response.json()
    cache_file.write_text(response.text)

    return data


def extract_total_tvl_series(protocol_data: Dict) -> Dict[int, float]:
    """Extract total TVL series across all chains for a protocol."""
    if not protocol_data:
        return {}

    tvl_series = {}

    # First try to use the 'tvl' key if available
    if "tvl" in protocol_data:
        for entry in protocol_data["tvl"]:
            timestamp = entry["date"]
            tvl = entry["totalLiquidityUSD"]
            tvl_series[timestamp] = tvl
        return tvl_series

    # If 'tvl' is missing, try to sum chainTvls
    if "chainTvls" in protocol_data:
        # Get all timestamps from chainTvls
        all_timestamps = set()
        for chain_data in protocol_data["chainTvls"].values():
            for entry in chain_data["tvl"]:
                all_timestamps.add(entry["date"])

        # For each timestamp, sum TVL across all chains
        for timestamp in all_timestamps:
            total_tvl = 0
            for chain_data in protocol_data["chainTvls"].values():
                # Find TVL for this timestamp in this chain
                for entry in chain_data["tvl"]:
                    if entry["date"] == timestamp:
                        total_tvl += entry["totalLiquidityUSD"]
                        break
            if total_tvl > 0:
                tvl_series[timestamp] = total_tvl

    return tvl_series


def forward_fill_tvl_data(
    tvl_series: Dict[int, float], max_gap_days: int = 7
) -> Dict[int, float]:
    """Forward-fill missing TVL data points to handle short outages."""
    if not tvl_series:
        return tvl_series

    timestamps = sorted(tvl_series.keys())
    if len(timestamps) < 2:
        return tvl_series

    filled_series = tvl_series.copy()

    # Fill gaps between data points
    for i in range(len(timestamps) - 1):
        current_time = timestamps[i]
        next_time = timestamps[i + 1]
        gap_days = (next_time - current_time) / 86400  # Convert seconds to days

        # If gap is small enough, forward-fill with current value
        if gap_days <= max_gap_days and gap_days > 1:
            current_value = tvl_series[current_time]
            # Fill daily gaps
            fill_time = current_time + 86400  # Start from next day
            while fill_time < next_time:
                filled_series[fill_time] = current_value
                fill_time += 86400

    return filled_series


def fetch_global_tvl_history() -> Dict[int, float]:
    """Fetch global DeFi TVL history from DeFiLlama."""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "global_tvl_history.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return {int(entry["date"]): float(entry["totalLiquidityUSD"]) for entry in data}

    print("Fetching global DeFi TVL history...")
    url = "https://api.llama.fi/charts"
    response = requests.get(url, timeout=30)

    if not response.ok:
        print(f"Failed to fetch global TVL: {response.status_code}")
        return {}

    data = response.json()
    cache_file.write_text(response.text)

    # Convert to timestamp -> tvl mapping
    global_tvl = {}
    for entry in data:
        timestamp = int(entry["date"])
        tvl = float(entry["totalLiquidityUSD"])
        global_tvl[timestamp] = tvl

    return global_tvl


def calculate_tvl_shares(
    protocol_tvls: Dict[str, Dict[int, float]],
    global_tvl: Dict[int, float],
    start_date: datetime,
) -> Dict[str, Dict[int, float]]:
    """Calculate each protocol's share of global DeFi TVL over time."""
    print("Calculating TVL shares using global DeFi TVL...")

    # Get all timestamps where we have data for at least one protocol or global TVL
    all_timestamps = set()
    for tvl_data in protocol_tvls.values():
        all_timestamps.update(tvl_data.keys())
    all_timestamps.update(global_tvl.keys())

    # Filter to analysis period
    start_timestamp = int(start_date.timestamp())
    analysis_timestamps = [t for t in all_timestamps if t >= start_timestamp]
    analysis_timestamps.sort()

    print(f"Found {len(analysis_timestamps)} timestamps in analysis period")

    tvl_shares = {}

    for timestamp in analysis_timestamps:
        # Get global TVL at this timestamp
        global_tvl_available_times = [t for t in global_tvl.keys() if t <= timestamp]
        if not global_tvl_available_times:
            continue

        latest_global_time = max(global_tvl_available_times)
        total_global_tvl = global_tvl[latest_global_time]

        if total_global_tvl <= 0:
            continue

        protocol_tvls_at_t = {}

        for protocol, tvl_data in protocol_tvls.items():
            # Use the most recent TVL value before or at this timestamp
            available_times = [t for t in tvl_data.keys() if t <= timestamp]
            if available_times:
                latest_time = max(available_times)
                tvl_value = tvl_data[latest_time]
                if tvl_value > 0:  # Only include positive TVL
                    protocol_tvls_at_t[protocol] = tvl_value

        # Calculate shares using global TVL
        if len(protocol_tvls_at_t) > 0:
            for protocol in protocol_tvls_at_t:
                if protocol not in tvl_shares:
                    tvl_shares[protocol] = {}
                tvl_shares[protocol][timestamp] = (
                    protocol_tvls_at_t[protocol] / total_global_tvl
                )

    # Print debug info
    for protocol in tvl_shares:
        print(f"{protocol}: {len(tvl_shares[protocol])} data points")

    return tvl_shares


def simulate_lp_pool(
    protocol: str,
    tvl_shares: Dict[int, float],
    start_timestamp: int,
    end_timestamp: int,
    initial_value: float = 1000,
) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]]]:
    """
    Simulate UP/DOWN token LP pool for one protocol over one period.

    UP token tracks protocol's TVL share performance.
    DOWN token is inverse (1 - UP price).

    Uses Uniswap V2 constant product formula: x * y = k

    Returns:
        - pool_values: Dict mapping timestamp to total pool value
        - external_holdings: Dict mapping timestamp to (external_up_tokens, external_down_tokens)
    """
    if protocol not in tvl_shares:
        return {}, {}

    protocol_data = tvl_shares[protocol]

    # Get TVL shares at start and throughout the period
    available_times = [
        t for t in protocol_data.keys() if start_timestamp <= t <= end_timestamp
    ]
    available_times.sort()

    if len(available_times) < 2:
        return {}, {}

    start_share = protocol_data[available_times[0]]
    if start_share <= 0:
        return {}, {}

    pool_values = {}
    external_holdings = {}

    # Initial setup: UP and DOWN tokens both start at $0.50
    # Initial pool has 1000 UP tokens and 1000 DOWN tokens
    initial_up_tokens = 1000.0
    initial_down_tokens = 1000.0
    k = initial_up_tokens * initial_down_tokens  # Constant product

    # Track external holdings (tokens removed from pool)
    external_up_tokens = 0.0
    external_down_tokens = 0.0

    # Calculate withdrawal timing
    period_duration = end_timestamp - start_timestamp
    withdrawal_timestamp = start_timestamp + (period_duration * WITHDRAWAL_TIMING_PCT)
    withdrawal_executed = False

    for timestamp in available_times:
        current_share = protocol_data[timestamp]

        # UP token price based on TVL share performance
        # If share doubles, UP goes to $1; if share halves, UP goes to $0.25
        share_ratio = current_share / start_share
        up_price = 0.5 * share_ratio
        up_price = min(0.99, max(0.01, up_price))  # Keep within bounds

        down_price = 1.0 - up_price

        # Check if we should execute withdrawal
        if (
            WITHDRAWAL_ENABLED
            and not withdrawal_executed
            and timestamp >= withdrawal_timestamp
        ):

            # Calculate current token amounts before withdrawal
            pre_withdrawal_up = np.sqrt(k * down_price / up_price)
            pre_withdrawal_down = np.sqrt(k * up_price / down_price)

            # Remove percentage of liquidity (proportional withdrawal)
            withdrawn_up = pre_withdrawal_up * WITHDRAWAL_AMOUNT_PCT
            withdrawn_down = pre_withdrawal_down * WITHDRAWAL_AMOUNT_PCT

            # Update external holdings
            external_up_tokens += withdrawn_up
            external_down_tokens += withdrawn_down

            # Update pool constant k (reduce by withdrawal amount squared)
            remaining_factor = 1.0 - WITHDRAWAL_AMOUNT_PCT
            k = k * (remaining_factor**2)

            withdrawal_executed = True

        # For constant product AMM: up_tokens * down_tokens = k
        # up_tokens * up_price = down_tokens * down_price (equal value)
        # up_tokens * up_price + down_tokens * down_price = total_pool_value

        # Solve: up_tokens = sqrt(k * down_price / up_price)
        #        down_tokens = sqrt(k * up_price / down_price)
        current_up_tokens = np.sqrt(k * down_price / up_price)
        current_down_tokens = np.sqrt(k * up_price / down_price)

        # Total pool value (only what's still in the pool)
        pool_value = current_up_tokens * up_price + current_down_tokens * down_price
        pool_values[timestamp] = pool_value

        # Track external holdings at each timestamp
        external_holdings[timestamp] = (external_up_tokens, external_down_tokens)

    return pool_values, external_holdings


def run_portfolio_simulation(
    protocols: List[str], tvl_shares: Dict[str, Dict[int, float]], start_date: datetime
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, Tuple[float, float]]]]:
    """Run portfolio simulation with overlapping 1-month periods starting every PERIOD_SPACING_DAYS days."""
    print(
        f"Running overlapping portfolio simulations (periods spaced {PERIOD_SPACING_DAYS} days apart)..."
    )

    portfolio_performances = {}
    all_external_holdings = {}

    # Generate overlapping 1-month periods starting every PERIOD_SPACING_DAYS days
    current_date = start_date
    end_analysis_date = start_date + timedelta(days=ANALYSIS_MONTHS * 30)

    # Ensure we have enough data for full 30-day periods
    # Stop starting new periods 30 days before the end
    max_start_date = end_analysis_date - timedelta(days=SIMULATION_PERIOD_DAYS)

    period_count = 0
    while current_date <= max_start_date:
        period_end = current_date + timedelta(days=SIMULATION_PERIOD_DAYS)
        period_key = f"period_{period_count:03d}_{current_date.strftime('%Y-%m-%d')}"

        if period_count % 30 == 0:  # Print every 30th period to reduce output
            print(f"Simulating period: {period_key}")

        start_timestamp = int(current_date.timestamp())
        end_timestamp = int(period_end.timestamp())

        # Simulate portfolio for this period
        portfolio_value = {}
        period_external_holdings = {}
        protocols_with_data_at_timestamp = (
            {}
        )  # Track how many protocols have data at each timestamp
        successful_protocols = 0

        for protocol in protocols:
            pool_values, external_holdings = simulate_lp_pool(
                protocol, tvl_shares, start_timestamp, end_timestamp
            )

            if pool_values:
                timestamps = sorted(pool_values.keys())
                if timestamps and len(timestamps) >= 2:
                    start_value = pool_values[timestamps[0]]
                    end_value = pool_values[timestamps[-1]]
                    if start_value > 0:
                        successful_protocols += 1

                        # Calculate total value including external holdings
                        for t in timestamps:
                            if t not in portfolio_value:
                                portfolio_value[t] = 0
                            if t not in period_external_holdings:
                                period_external_holdings[t] = {}
                            if t not in protocols_with_data_at_timestamp:
                                protocols_with_data_at_timestamp[t] = 0

                            # Store external holdings for this protocol at this timestamp
                            period_external_holdings[t][protocol] = external_holdings[t]

                            # Calculate current token prices
                            if protocol in tvl_shares and t in tvl_shares[protocol]:
                                current_share = tvl_shares[protocol][t]
                                start_share = tvl_shares[protocol][timestamps[0]]
                                share_ratio = current_share / start_share
                                up_price = min(0.99, max(0.01, 0.5 * share_ratio))
                                down_price = 1.0 - up_price

                                # Value of external holdings
                                ext_up, ext_down = external_holdings[t]
                                external_value = (
                                    ext_up * up_price + ext_down * down_price
                                )

                                # Total value = pool value + external value
                                total_value = pool_values[t] + external_value

                                # Each protocol gets equal weight (1/N of portfolio)
                                protocol_return = (
                                    total_value / (start_value + 0) - 1
                                )  # start_value is initial pool value
                                portfolio_value[t] += protocol_return
                                protocols_with_data_at_timestamp[t] += 1

                        # Debug output for first few periods only
                        if period_count < 3:
                            final_timestamp = timestamps[-1]
                            ext_up, ext_down = external_holdings[final_timestamp]
                            final_share = tvl_shares[protocol][final_timestamp]
                            start_share = tvl_shares[protocol][timestamps[0]]
                            share_ratio = final_share / start_share
                            up_price = min(0.99, max(0.01, 0.5 * share_ratio))
                            down_price = 1.0 - up_price
                            external_value = ext_up * up_price + ext_down * down_price
                            total_final_value = end_value + external_value

                            print(
                                f"  {protocol}: pool={end_value:.2f}, external={external_value:.2f}, "
                                f"total={total_final_value:.2f}, return={((total_final_value/start_value-1)*100):.2f}%"
                            )

        # Average across protocols that actually have data at each timestamp
        for t in portfolio_value:
            if protocols_with_data_at_timestamp.get(t, 0) > 0:
                portfolio_value[t] /= protocols_with_data_at_timestamp[t]

        if (
            period_count < 3 and portfolio_value
        ):  # Only print detailed info for first few periods
            final_return = (
                portfolio_value[max(portfolio_value.keys())] if portfolio_value else 0
            )
            print(
                f"  Period summary: {successful_protocols} protocols, "
                f"portfolio return: {(final_return*100):.2f}%"
            )
        elif period_count < 3:
            print(f"  No successful protocols in {period_key}")

        portfolio_performances[period_key] = portfolio_value
        all_external_holdings[period_key] = period_external_holdings

        # Move to next period start
        current_date = current_date + timedelta(days=PERIOD_SPACING_DAYS)
        period_count += 1

    print(f"Generated {period_count} overlapping 1-month periods")
    return portfolio_performances, all_external_holdings


def apply_interest_rate_discount(
    portfolio_performances: Dict[str, Dict[int, float]]
) -> Dict[str, Dict[int, float]]:
    """Apply daily interest rate discounting to portfolio returns."""
    print("Applying interest rate discounting...")

    daily_rate = (1 + ANNUAL_INTEREST_RATE) ** (1 / 365) - 1
    discounted_performances = {}

    for period, performance in portfolio_performances.items():
        discounted_performance = {}
        timestamps = sorted(performance.keys())

        if not timestamps:
            continue

        start_timestamp = timestamps[0]

        for timestamp in timestamps:
            days_elapsed = (timestamp - start_timestamp) / 86400
            discount_factor = (1 + daily_rate) ** days_elapsed

            # Adjust return for opportunity cost
            raw_return = performance[timestamp]
            discounted_return = (1 + raw_return) / discount_factor - 1
            discounted_performance[timestamp] = discounted_return

        discounted_performances[period] = discounted_performance

    return discounted_performances


def plot_portfolio_performance(portfolio_performances: Dict[str, Dict[int, float]]):
    """Create overlay plot of all portfolio performance periods."""
    print("Creating performance plot...")

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(portfolio_performances)))

    all_final_returns = []

    for i, (_, performance) in enumerate(portfolio_performances.items()):
        if not performance:
            continue

        timestamps = sorted(performance.keys())
        if len(timestamps) < 2:
            continue

        # Convert to days from start of period
        start_timestamp = timestamps[0]
        days = [(t - start_timestamp) / 86400 for t in timestamps]
        returns = [performance[t] * 100 for t in timestamps]  # Convert to percentage

        plt.plot(days, returns, color=colors[i], alpha=0.7, linewidth=1)

        if returns:
            all_final_returns.append(returns[-1])

    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    plt.xlabel("Days into Period")
    plt.ylabel("Portfolio Return (%)")
    withdrawal_text = (
        f"Withdrawal: {WITHDRAWAL_AMOUNT_PCT:.0%} at {WITHDRAWAL_TIMING_PCT:.0%}"
        if WITHDRAWAL_ENABLED
        else "No Withdrawal"
    )
    plt.title(
        f"UP/DOWN Token LP Portfolio Performance (Overlapping 1-Month Periods)\n"
        f"(Top {TOP_N_PROTOCOLS} DeFi Protocols, {ANNUAL_INTEREST_RATE:.0%} Annual Discount Rate, {withdrawal_text})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_dir = Path("portfolio_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(
        output_dir / "portfolio_performance_overlay.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    return all_final_returns


def calculate_performance_statistics(
    all_final_returns: List[float],
) -> Dict[str, float]:
    """Calculate performance statistics."""
    if not all_final_returns:
        return {}

    returns_array = np.array(all_final_returns)

    stats = {
        "mean_return_pct": float(np.mean(returns_array)),
        "median_return_pct": float(np.median(returns_array)),
        "std_return_pct": float(np.std(returns_array)),
        "min_return_pct": float(np.min(returns_array)),
        "max_return_pct": float(np.max(returns_array)),
        "percentile_10_worst": float(np.percentile(returns_array, 10)),
        "percentile_20_worst": float(np.percentile(returns_array, 20)),
        "percentile_40_worst": float(np.percentile(returns_array, 40)),
        "percentile_60_worst": float(np.percentile(returns_array, 60)),
        "percentile_80_worst": float(np.percentile(returns_array, 80)),
        "percentile_90_worst": float(np.percentile(returns_array, 90)),
        "positive_periods": int(np.sum(returns_array > 0)),
        "total_periods": len(returns_array),
    }

    return stats


def main():
    """Main execution function."""
    print("=== DeFi LP Portfolio Simulation ===\n")

    # 1. Fetch top protocols
    top_protocols = fetch_top_protocols(TOP_N_PROTOCOLS)
    protocol_slugs = [p["slug"] for p in top_protocols]

    # 2. Fetch historical data
    print("\nFetching historical TVL data...")
    protocol_tvls = {}
    for slug in protocol_slugs:
        print(f"Fetching data for {slug}...")
        protocol_data = fetch_protocol_history(slug)
        tvl_series = extract_total_tvl_series(protocol_data)
        if tvl_series:
            # Apply forward-fill to handle missing data points
            filled_series = forward_fill_tvl_data(tvl_series)
            protocol_tvls[slug] = filled_series

    print(f"Successfully fetched data for {len(protocol_tvls)} protocols")

    # 3. Fetch global TVL history
    global_tvl = fetch_global_tvl_history()
    if not global_tvl:
        print(
            "Warning: Failed to fetch global TVL data, calculations may be inaccurate"
        )

    # 4. Calculate TVL shares using global TVL
    start_date = datetime.now() - timedelta(days=ANALYSIS_MONTHS * 30)
    tvl_shares = calculate_tvl_shares(protocol_tvls, global_tvl, start_date)

    # 5. Run portfolio simulation
    portfolio_performances, _ = run_portfolio_simulation(
        list(protocol_tvls.keys()), tvl_shares, start_date
    )

    # 6. Apply interest rate discounting
    discounted_performances = apply_interest_rate_discount(portfolio_performances)

    # 7. Plot results
    final_returns = plot_portfolio_performance(discounted_performances)

    # 8. Calculate and save statistics
    stats = calculate_performance_statistics(final_returns)

    print("\n=== PERFORMANCE STATISTICS ===")
    print(f"Mean Return: {stats.get('mean_return_pct', 0):.2f}%")
    print(f"Median Return: {stats.get('median_return_pct', 0):.2f}%")
    # print(f"Standard Deviation: {stats.get('std_return_pct', 0):.2f}%")
    print(f"10th Percentile (worst): {stats.get('percentile_10_worst', 0):.2f}%")
    print(f"20th Percentile: {stats.get('percentile_20_worst', 0):.2f}%")
    print(f"40th Percentile: {stats.get('percentile_40_worst', 0):.2f}%")
    print(f"60th Percentile: {stats.get('percentile_60_worst', 0):.2f}%")
    print(f"80th Percentile: {stats.get('percentile_80_worst', 0):.2f}%")
    print(f"90th Percentile: {stats.get('percentile_90_worst', 0):.2f}%")
    print(
        f"Positive Periods: {stats.get('positive_periods', 0)}/{stats.get('total_periods', 0)}"
    )

    # Save results
    output_dir = Path("portfolio_results")
    output_dir.mkdir(exist_ok=True)

    results = {
        "configuration": {
            "top_n_protocols": TOP_N_PROTOCOLS,
            "analysis_months": ANALYSIS_MONTHS,
            "simulation_period_days": SIMULATION_PERIOD_DAYS,
            "annual_interest_rate": ANNUAL_INTEREST_RATE,
            "withdrawal_enabled": WITHDRAWAL_ENABLED,
            "withdrawal_timing_pct": WITHDRAWAL_TIMING_PCT,
            "withdrawal_amount_pct": WITHDRAWAL_AMOUNT_PCT,
        },
        "protocols_analyzed": list(protocol_tvls.keys()),
        "statistics": stats,
        "final_returns": final_returns,
    }

    with open(output_dir / "simulation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to portfolio_results/")
    print("- simulation_results.json: Detailed statistics and data")
    print("- portfolio_performance_overlay.png: Performance visualization")


if __name__ == "__main__":
    main()
