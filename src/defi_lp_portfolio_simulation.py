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

    # Filter for onchain DeFi protocols (exclude CEXes and staking)
    excluded_categories = {
        "CEX",
        "Liquid Staking",
        "Staking",
        "Restaking",
        "Bridge",
        "Liquid Restaking",
        "Canonical Bridge",
    }
    defi_protocols = [
        p for p in protocols if p.get("category") not in excluded_categories
    ]
    defi_protocols.sort(key=lambda x: x.get("tvl", 0) or 0, reverse=True)

    if len(defi_protocols) < n:
        return defi_protocols

    top_protocols = defi_protocols[:n]
    print(f"Selected protocols: {[p['name'] for p in top_protocols]}")

    return top_protocols


def cached_api_fetch(url: str, cache_file: Path) -> Dict:
    """Generic function to fetch data from API with caching."""
    cache_file.parent.mkdir(exist_ok=True)

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    response = requests.get(url, timeout=30)
    if not response.ok:
        print(f"Failed to fetch {url}: {response.status_code}")
        return {}

    cache_file.write_text(response.text)
    return response.json()


def fetch_protocol_history(slug: str) -> Dict:
    """Fetch historical TVL data for a protocol."""
    cache_file = Path("cache") / f"{slug}_history.json"
    return cached_api_fetch(f"https://api.llama.fi/protocol/{slug}", cache_file)


def extract_chain_tvl_series(chain_tvls: Dict) -> Dict[int, float]:
    """Sum TVL across all chains for each timestamp."""
    all_timestamps = set()
    for chain_data in chain_tvls.values():
        all_timestamps.update(entry["date"] for entry in chain_data["tvl"])

    tvl_series = {}
    for timestamp in all_timestamps:
        total_tvl = sum(
            entry["totalLiquidityUSD"]
            for chain_data in chain_tvls.values()
            for entry in chain_data["tvl"]
            if entry["date"] == timestamp
        )
        if total_tvl > 0:
            tvl_series[timestamp] = total_tvl

    return tvl_series


def extract_total_tvl_series(protocol_data: Dict) -> Dict[int, float]:
    """Extract total TVL series across all chains for a protocol."""
    if not protocol_data:
        return {}

    # Try to use the 'tvl' key first
    if "tvl" in protocol_data:
        return {
            entry["date"]: entry.get("tvl", entry.get("totalLiquidityUSD", 0))
            for entry in protocol_data["tvl"]
        }

    # Fall back to summing chainTvls
    if "chainTvls" in protocol_data:
        return extract_chain_tvl_series(protocol_data["chainTvls"])

    return {}


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
            # Fill daily gaps with tolerance for DST shifts
            fill_time = current_time + 86400  # Start from next day
            while fill_time < next_time:
                # Check for existing timestamp within ±1 hour (DST tolerance)
                existing_timestamp = None
                for existing_time in tvl_series.keys():
                    if abs(existing_time - fill_time) <= 3600:  # Within 1 hour
                        existing_timestamp = existing_time
                        break

                if not existing_timestamp:
                    filled_series[fill_time] = current_value
                fill_time += 86400

    return filled_series


def find_latest_timestamp(timestamps: List[int], target: int) -> int:
    """Find the latest timestamp <= target from a list of timestamps."""
    available = [t for t in timestamps if t <= target]
    return max(available) if available else None


def fetch_global_tvl_history() -> Dict[int, float]:
    """Fetch global DeFi TVL history from DeFiLlama (excludes liquid staking and double counted TVL)."""
    print("Fetching global DeFi TVL history...")
    cache_file = Path("cache") / "global_tvl_history.json"
    data = cached_api_fetch("https://api.llama.fi/v2/historicalChainTvl", cache_file)

    if not data:
        return {}

    return {int(entry["date"]): float(entry["tvl"]) for entry in data}


def get_tvl_at_timestamp(tvl_data: Dict[int, float], timestamp: int) -> float:
    """Get TVL value at or before the given timestamp."""
    latest_time = find_latest_timestamp(list(tvl_data.keys()), timestamp)
    return tvl_data[latest_time] if latest_time else 0.0


def calculate_protocol_share_at_timestamp(
    protocol_tvls: Dict[str, Dict[int, float]],
    global_tvl: Dict[int, float],
    timestamp: int,
) -> Dict[str, float]:
    """Calculate each protocol's share at a specific timestamp."""
    total_global_tvl = get_tvl_at_timestamp(global_tvl, timestamp)
    if total_global_tvl <= 0:
        return {}

    return {
        protocol: max(0, get_tvl_at_timestamp(tvl_data, timestamp) / total_global_tvl)
        for protocol, tvl_data in protocol_tvls.items()
    }


def calculate_tvl_shares(
    protocol_tvls: Dict[str, Dict[int, float]],
    global_tvl: Dict[int, float],
    start_date: datetime,
) -> Dict[str, Dict[int, float]]:
    """Calculate each protocol's share of global DeFi TVL over time."""
    print("Calculating TVL shares using global DeFi TVL...")

    # Get analysis timestamps
    all_timestamps = set()
    for tvl_data in protocol_tvls.values():
        all_timestamps.update(tvl_data.keys())
    all_timestamps.update(global_tvl.keys())

    start_timestamp = int(start_date.timestamp())
    analysis_timestamps = sorted(t for t in all_timestamps if t >= start_timestamp)
    print(f"Found {len(analysis_timestamps)} timestamps in analysis period")

    # Initialize shares for all protocols
    tvl_shares = {protocol: {} for protocol in protocol_tvls.keys()}

    for timestamp in analysis_timestamps:
        shares_at_t = calculate_protocol_share_at_timestamp(
            protocol_tvls, global_tvl, timestamp
        )

        for protocol in protocol_tvls.keys():
            if protocol in shares_at_t:
                tvl_shares[protocol][timestamp] = shares_at_t[protocol]
            else:
                # Forward-fill from previous timestamp
                prev_time = find_latest_timestamp(
                    list(tvl_shares[protocol].keys()), timestamp - 1
                )
                tvl_shares[protocol][timestamp] = (
                    tvl_shares[protocol][prev_time] if prev_time else 0.0
                )

    # Print debug info
    for protocol in tvl_shares:
        print(f"{protocol}: {len(tvl_shares[protocol])} data points")

    return tvl_shares


def calculate_token_prices(share_ratio: float) -> Tuple[float, float]:
    """Calculate UP and DOWN token prices based on TVL share performance."""
    up_price = min(0.99, max(0.01, 0.5 * share_ratio))
    return up_price, 1.0 - up_price


def execute_withdrawal(
    k: float, up_price: float, down_price: float
) -> Tuple[float, float, float]:
    """Execute proportional withdrawal from LP pool."""
    pre_withdrawal_up = np.sqrt(k * down_price / up_price)
    pre_withdrawal_down = np.sqrt(k * up_price / down_price)

    withdrawn_up = pre_withdrawal_up * WITHDRAWAL_AMOUNT_PCT
    withdrawn_down = pre_withdrawal_down * WITHDRAWAL_AMOUNT_PCT

    remaining_factor = 1.0 - WITHDRAWAL_AMOUNT_PCT
    new_k = k * (remaining_factor**2)

    return withdrawn_up, withdrawn_down, new_k


def calculate_pool_value(k: float, up_price: float, down_price: float) -> float:
    """Calculate total pool value using constant product formula."""
    up_tokens = np.sqrt(k * down_price / up_price)
    down_tokens = np.sqrt(k * up_price / down_price)
    return up_tokens * up_price + down_tokens * down_price


def simulate_lp_pool(
    protocol: str,
    tvl_shares: Dict[int, float],
    start_timestamp: int,
    end_timestamp: int,
    initial_value: float = 1000,
) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]]]:
    """Simulate UP/DOWN token LP pool for one protocol over one period."""
    if protocol not in tvl_shares:
        return {}, {}

    protocol_data = tvl_shares[protocol]
    available_times = [
        t for t in protocol_data.keys() if start_timestamp <= t <= end_timestamp
    ]
    available_times.sort()

    if len(available_times) < 2:
        return {}, {}

    start_share = protocol_data[available_times[0]]
    if start_share <= 0:
        return {}, {}

    # Initialize pool state
    k = 1000.0 * 1000.0  # Initial constant product (1000 UP * 1000 DOWN tokens)
    external_up_tokens, external_down_tokens = 0.0, 0.0
    withdrawal_executed = False

    # Calculate withdrawal timing
    period_duration = end_timestamp - start_timestamp
    withdrawal_timestamp = start_timestamp + (period_duration * WITHDRAWAL_TIMING_PCT)

    pool_values = {}
    external_holdings = {}

    for timestamp in available_times:
        current_share = protocol_data[timestamp]
        share_ratio = current_share / start_share
        up_price, down_price = calculate_token_prices(share_ratio)

        # Execute withdrawal if needed
        if (
            WITHDRAWAL_ENABLED
            and not withdrawal_executed
            and timestamp >= withdrawal_timestamp
        ):
            withdrawn_up, withdrawn_down, k = execute_withdrawal(
                k, up_price, down_price
            )
            external_up_tokens += withdrawn_up
            external_down_tokens += withdrawn_down
            withdrawal_executed = True

        # Calculate pool value and track external holdings
        pool_values[timestamp] = calculate_pool_value(k, up_price, down_price)
        external_holdings[timestamp] = (external_up_tokens, external_down_tokens)

    return pool_values, external_holdings


def calculate_protocol_contribution(
    protocol: str,
    pool_values: Dict[int, float],
    external_holdings: Dict[int, Tuple[float, float]],
    tvl_shares: Dict[str, Dict[int, float]],
    timestamps: List[int],
) -> Dict[int, float]:
    """Calculate a protocol's contribution to portfolio returns over time."""
    contributions = {}
    start_share = tvl_shares[protocol][timestamps[0]]

    for t in timestamps:
        if t in pool_values and protocol in tvl_shares and t in tvl_shares[protocol]:
            # Calculate token prices at this timestamp
            current_share = tvl_shares[protocol][t]
            share_ratio = current_share / start_share
            up_price, down_price = calculate_token_prices(share_ratio)

            # Value of external holdings
            ext_up, ext_down = external_holdings[t]
            external_value = ext_up * up_price + ext_down * down_price

            # Total value = pool value + external value
            total_value = pool_values[t] + external_value

            # Calculate return (each protocol starts with 1000 investment)
            protocol_return = total_value / 1000.0 - 1
            contributions[t] = protocol_return

    return contributions


def simulate_single_period(
    protocols: List[str],
    tvl_shares: Dict[str, Dict[int, float]],
    start_timestamp: int,
    end_timestamp: int,
    period_key: str,
    debug: bool = False,
) -> Dict[int, float]:
    """Simulate portfolio for a single period."""
    portfolio_value = {}
    protocols_with_data_at_timestamp = {}
    successful_protocols = 0

    for protocol in protocols:
        pool_values, external_holdings = simulate_lp_pool(
            protocol, tvl_shares, start_timestamp, end_timestamp
        )

        if not pool_values:
            continue

        timestamps = sorted(pool_values.keys())
        if len(timestamps) < 2 or pool_values[timestamps[0]] <= 0:
            continue

        successful_protocols += 1
        contributions = calculate_protocol_contribution(
            protocol, pool_values, external_holdings, tvl_shares, timestamps
        )

        # Add to portfolio
        for t, contribution in contributions.items():
            portfolio_value[t] = portfolio_value.get(t, 0) + contribution
            protocols_with_data_at_timestamp[t] = (
                protocols_with_data_at_timestamp.get(t, 0) + 1
            )

        # Debug output for first few periods
        if debug and successful_protocols <= 3:
            final_value = pool_values[timestamps[-1]] + sum(
                ext * price
                for ext, price in zip(
                    external_holdings[timestamps[-1]],
                    calculate_token_prices(
                        tvl_shares[protocol][timestamps[-1]]
                        / tvl_shares[protocol][timestamps[0]]
                    ),
                )
            )
            print(
                f"  {protocol}: total={final_value:.2f}, return={((final_value/1000-1)*100):.2f}%"
            )

    # Average across protocols that have data at each timestamp
    for t in portfolio_value:
        if protocols_with_data_at_timestamp.get(t, 0) > 0:
            portfolio_value[t] /= protocols_with_data_at_timestamp[t]

    if debug and portfolio_value:
        final_return = portfolio_value[max(portfolio_value.keys())]
        print(
            f"  Period summary: {successful_protocols} protocols, portfolio return: {(final_return*100):.2f}%"
        )
    elif debug:
        print(f"  No successful protocols in {period_key}")

    return portfolio_value


def run_portfolio_simulation(
    protocols: List[str], tvl_shares: Dict[str, Dict[int, float]], start_date: datetime
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, Tuple[float, float]]]]:
    """Run portfolio simulation with overlapping 1-month periods."""
    print(
        f"Running overlapping portfolio simulations (periods spaced {PERIOD_SPACING_DAYS} days apart)..."
    )

    portfolio_performances = {}
    all_external_holdings = {}

    # Generate date ranges for overlapping periods
    current_date = start_date
    end_analysis_date = start_date + timedelta(days=ANALYSIS_MONTHS * 30)
    max_start_date = end_analysis_date - timedelta(days=SIMULATION_PERIOD_DAYS)

    period_count = 0
    while current_date <= max_start_date:
        period_end = current_date + timedelta(days=SIMULATION_PERIOD_DAYS)
        period_key = f"period_{period_count:03d}_{current_date.strftime('%Y-%m-%d')}"

        if period_count % 30 == 0:
            print(f"Simulating period: {period_key}")

        start_timestamp = int(current_date.timestamp())
        end_timestamp = int(period_end.timestamp())

        # Simulate this period
        portfolio_value = simulate_single_period(
            protocols,
            tvl_shares,
            start_timestamp,
            end_timestamp,
            period_key,
            debug=(period_count < 3),
        )

        portfolio_performances[period_key] = portfolio_value
        all_external_holdings[period_key] = (
            {}
        )  # Simplified - not tracking detailed external holdings

        current_date += timedelta(days=PERIOD_SPACING_DAYS)
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
            # portfolio_value contains period-to-date returns (compounded)
            raw_return = performance[timestamp]
            # Convert to value, discount, then back to return
            portfolio_value_at_t = 1 + raw_return  # Convert return to value
            discounted_value = portfolio_value_at_t / discount_factor
            discounted_return = discounted_value - 1
            discounted_performance[timestamp] = discounted_return

        discounted_performances[period] = discounted_performance

    return discounted_performances


def to_percentage(decimal_values: List[float]) -> List[float]:
    """Convert decimal returns to percentage values."""
    return [r * 100 for r in decimal_values]


def plot_portfolio_performance(
    portfolio_performances: Dict[str, Dict[int, float]]
) -> List[float]:
    """Create overlay plot of all portfolio performance periods."""
    print("Creating performance plot...")

    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(portfolio_performances)))
    all_final_returns = []

    for i, (_, performance) in enumerate(portfolio_performances.items()):
        if not performance or len(performance) < 2:
            continue

        timestamps = sorted(performance.keys())
        start_timestamp = timestamps[0]

        # Convert to days and percentage returns for plotting
        days = [(t - start_timestamp) / 86400 for t in timestamps]
        returns = [performance[t] for t in timestamps]

        plt.plot(days, to_percentage(returns), color=colors[i], alpha=0.7, linewidth=1)
        all_final_returns.append(returns[-1])  # Keep as decimal for statistics

    # Format plot
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


def calculate_percentiles(returns_array: np.ndarray) -> Dict[str, float]:
    """Calculate percentile statistics for returns."""
    percentiles = [10, 20, 40, 60, 80, 90]
    return {
        f"percentile_{p}_worst": float(np.percentile(returns_array, p) * 100)
        for p in percentiles
    }


def calculate_performance_statistics(
    all_final_returns: List[float],
) -> Dict[str, float]:
    """Calculate performance statistics."""
    if not all_final_returns:
        return {}

    returns_array = np.array(all_final_returns)
    returns_pct = returns_array * 100

    stats = {
        "mean_return_pct": float(np.mean(returns_pct)),
        "median_return_pct": float(np.median(returns_pct)),
        "std_return_pct": float(np.std(returns_pct)),
        "min_return_pct": float(np.min(returns_pct)),
        "max_return_pct": float(np.max(returns_pct)),
        "positive_periods": int(np.sum(returns_array > 0)),
        "total_periods": len(returns_array),
    }

    # Add percentiles
    stats.update(calculate_percentiles(returns_array))

    return stats


def main() -> None:
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
