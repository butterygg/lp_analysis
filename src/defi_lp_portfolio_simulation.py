#!/usr/bin/env python3
"""
LP Portfolio Simulation Script

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
TOP_EVM_CHAINS = [
    "Ethereum",
    "BSC",
    "Base",
    "Arbitrum",
    "Avalanche",
]  # Top 5 EVM chains by TVL
ANALYSIS_MONTHS = 24
SIMULATION_PERIOD_DAYS = 30  # One month simulation periods
PERIOD_SPACING_DAYS = 1  # Number of days between the start of each period
DAILY_FEE_RATE = 0.003  # 0.3% daily trading fees (typical for AMM pools)

# Liquidity withdrawal configuration
WITHDRAWAL_ENABLED = True  # Enable partial liquidity withdrawal
WITHDRAWAL_TIMING_PCT = 0.25  # When to withdraw (0.5 = halfway through period)
WITHDRAWAL_AMOUNT_PCT = 0.7  # How much to withdraw (0.5 = 50% of liquidity)


def get_top_evm_chains() -> List[str]:
    """Return the hardcoded list of top EVM chains."""
    print(f"Using top {len(TOP_EVM_CHAINS)} EVM chains: {', '.join(TOP_EVM_CHAINS)}")
    return TOP_EVM_CHAINS


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


def fetch_chain_history(chain: str) -> Dict[int, float]:
    """Fetch historical TVL data for a chain."""
    cache_file = Path("cache") / f"chain_{chain}_history.json"
    url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
    data = cached_api_fetch(url, cache_file)

    if not data:
        return {}

    return {int(entry["date"]): float(entry["tvl"]) for entry in data}


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


def calculate_chain_share_at_timestamp(
    chain_tvls: Dict[str, Dict[int, float]],
    global_tvl: Dict[int, float],
    timestamp: int,
) -> Dict[str, float]:
    """Calculate each chain's share at a specific timestamp."""
    total_global_tvl = get_tvl_at_timestamp(global_tvl, timestamp)
    if total_global_tvl <= 0:
        return {}

    return {
        chain: max(0, get_tvl_at_timestamp(tvl_data, timestamp) / total_global_tvl)
        for chain, tvl_data in chain_tvls.items()
    }


def calculate_tvl_shares(
    chain_tvls: Dict[str, Dict[int, float]],
    global_tvl: Dict[int, float],
    start_date: datetime,
) -> Dict[str, Dict[int, float]]:
    """Calculate each chain's share of global DeFi TVL over time."""
    print("Calculating TVL shares per chain using global DeFi TVL...")

    # Get analysis timestamps
    all_timestamps = set()
    for tvl_data in chain_tvls.values():
        all_timestamps.update(tvl_data.keys())
    all_timestamps.update(global_tvl.keys())

    start_timestamp = int(start_date.timestamp())
    analysis_timestamps = sorted(t for t in all_timestamps if t >= start_timestamp)
    print(f"Found {len(analysis_timestamps)} timestamps in analysis period")

    # Initialize shares for all chains
    tvl_shares = {chain: {} for chain in chain_tvls.keys()}

    for timestamp in analysis_timestamps:
        shares_at_t = calculate_chain_share_at_timestamp(
            chain_tvls, global_tvl, timestamp
        )

        for chain in chain_tvls.keys():
            if chain in shares_at_t:
                tvl_shares[chain][timestamp] = shares_at_t[chain]
            else:
                # Forward-fill from previous timestamp
                prev_time = find_latest_timestamp(
                    list(tvl_shares[chain].keys()), timestamp - 1
                )
                tvl_shares[chain][timestamp] = (
                    tvl_shares[chain][prev_time] if prev_time else 0.0
                )

    # Print debug info
    for chain in tvl_shares:
        print(f"{chain}: {len(tvl_shares[chain])} data points")

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
    chain: str,
    tvl_shares: Dict[int, float],
    start_timestamp: int,
    end_timestamp: int,
    initial_value: float = 1000,
) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]], float]:
    """Simulate UP/DOWN token LP pool for one chain over one period."""
    if chain not in tvl_shares:
        return {}, {}, 0.0

    chain_data = tvl_shares[chain]
    available_times = [
        t for t in chain_data.keys() if start_timestamp <= t <= end_timestamp
    ]
    available_times.sort()

    if len(available_times) < 2:
        return {}, {}, 0.0

    start_share = chain_data[available_times[0]]
    if start_share <= 0:
        return {}, {}, 0.0

    # Initialize pool state
    k = 1000.0 * 1000.0  # Initial constant product (1000 UP * 1000 DOWN tokens)
    external_up_tokens, external_down_tokens = 0.0, 0.0
    withdrawal_executed = False
    accumulated_fees = 0.0
    last_timestamp = available_times[0]

    # Calculate withdrawal timing
    period_duration = end_timestamp - start_timestamp
    withdrawal_timestamp = start_timestamp + (period_duration * WITHDRAWAL_TIMING_PCT)

    pool_values = {}
    external_holdings = {}

    for timestamp in available_times:
        current_share = chain_data[timestamp]
        share_ratio = current_share / start_share
        up_price, down_price = calculate_token_prices(share_ratio)

        # Calculate fees based on days elapsed since last update
        days_elapsed = (timestamp - last_timestamp) / 86400
        if days_elapsed > 0:
            # Fee accrual based on current pool value
            current_pool_value = calculate_pool_value(k, up_price, down_price)
            daily_fees = current_pool_value * DAILY_FEE_RATE * days_elapsed
            accumulated_fees += daily_fees

        last_timestamp = timestamp

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

    return pool_values, external_holdings, accumulated_fees


def calculate_chain_contribution(
    chain: str,
    pool_values: Dict[int, float],
    external_holdings: Dict[int, Tuple[float, float]],
    tvl_shares: Dict[str, Dict[int, float]],
    timestamps: List[int],
    fees: float,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Calculate a chain's contribution to portfolio returns over time.
    Returns: (total_contributions, fee_contributions, il_contributions)
    """
    total_contributions = {}
    fee_contributions = {}
    il_contributions = {}
    start_share = tvl_shares[chain][timestamps[0]]

    for t in timestamps:
        if t in pool_values and chain in tvl_shares and t in tvl_shares[chain]:
            # Calculate token prices at this timestamp
            current_share = tvl_shares[chain][t]
            share_ratio = current_share / start_share
            up_price, down_price = calculate_token_prices(share_ratio)

            # Value of external holdings
            ext_up, ext_down = external_holdings[t]
            external_value = ext_up * up_price + ext_down * down_price

            # Total value = pool value + external value + accumulated fees
            total_value = pool_values[t] + external_value + fees

            # Calculate return components (each chain starts with 1000 investment)
            total_return = total_value / 1000.0 - 1
            fee_return = fees / 1000.0
            # IL return is everything except fees: (pool + external - initial) / initial
            portfolio_value_no_fees = pool_values[t] + external_value
            il_return = portfolio_value_no_fees / 1000.0 - 1

            total_contributions[t] = total_return
            fee_contributions[t] = fee_return
            il_contributions[t] = il_return

    return total_contributions, fee_contributions, il_contributions


def simulate_single_period(
    chains: List[str],
    tvl_shares: Dict[str, Dict[int, float]],
    start_timestamp: int,
    end_timestamp: int,
    period_key: str,
    debug: bool = False,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Simulate portfolio for a single period.
    Returns: (total_portfolio_value, fee_portfolio_value, il_portfolio_value)
    """
    portfolio_value = {}
    fee_portfolio_value = {}
    il_portfolio_value = {}
    chains_with_data_at_timestamp = {}
    successful_chains = 0

    for chain in chains:
        pool_values, external_holdings, fees = simulate_lp_pool(
            chain, tvl_shares, start_timestamp, end_timestamp
        )

        if not pool_values:
            continue

        timestamps = sorted(pool_values.keys())
        if len(timestamps) < 2 or pool_values[timestamps[0]] <= 0:
            continue

        successful_chains += 1
        total_contributions, fee_contributions, il_contributions = (
            calculate_chain_contribution(
                chain, pool_values, external_holdings, tvl_shares, timestamps, fees
            )
        )

        # Add to portfolio
        for t in total_contributions:
            portfolio_value[t] = portfolio_value.get(t, 0) + total_contributions[t]
            fee_portfolio_value[t] = (
                fee_portfolio_value.get(t, 0) + fee_contributions[t]
            )
            il_portfolio_value[t] = il_portfolio_value.get(t, 0) + il_contributions[t]
            chains_with_data_at_timestamp[t] = (
                chains_with_data_at_timestamp.get(t, 0) + 1
            )

        # Debug output for first few periods
        if debug and successful_chains <= 3:
            final_value = (
                pool_values[timestamps[-1]]
                + fees
                + sum(
                    ext * price
                    for ext, price in zip(
                        external_holdings[timestamps[-1]],
                        calculate_token_prices(
                            tvl_shares[chain][timestamps[-1]]
                            / tvl_shares[chain][timestamps[0]]
                        ),
                    )
                )
            )
            print(
                f"  {chain}: total={final_value:.2f}, return={((final_value/1000-1)*100):.2f}%"
            )

    # Average across chains that have data at each timestamp
    for t in portfolio_value:
        if chains_with_data_at_timestamp.get(t, 0) > 0:
            portfolio_value[t] /= chains_with_data_at_timestamp[t]
            fee_portfolio_value[t] /= chains_with_data_at_timestamp[t]
            il_portfolio_value[t] /= chains_with_data_at_timestamp[t]

    if debug and portfolio_value:
        final_return = portfolio_value[max(portfolio_value.keys())]
        print(
            f"  Period summary: {successful_chains} chains, portfolio return: {(final_return*100):.2f}%"
        )
    elif debug:
        print(f"  No successful chains in {period_key}")

    return portfolio_value, fee_portfolio_value, il_portfolio_value


def run_portfolio_simulation(
    chains: List[str], tvl_shares: Dict[str, Dict[int, float]], start_date: datetime
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
]:
    """Run portfolio simulation with overlapping 1-month periods.
    Returns: (total_performances, fee_performances, il_performances)
    """
    print(
        f"Running overlapping portfolio simulations (periods spaced {PERIOD_SPACING_DAYS} days apart)..."
    )

    portfolio_performances = {}
    fee_performances = {}
    il_performances = {}

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
        portfolio_value, fee_value, il_value = simulate_single_period(
            chains,
            tvl_shares,
            start_timestamp,
            end_timestamp,
            period_key,
            debug=(period_count < 3),
        )

        portfolio_performances[period_key] = portfolio_value
        fee_performances[period_key] = fee_value
        il_performances[period_key] = il_value

        current_date += timedelta(days=PERIOD_SPACING_DAYS)
        period_count += 1

    print(f"Generated {period_count} overlapping 1-month periods")
    return portfolio_performances, fee_performances, il_performances


def to_percentage(decimal_values: List[float]) -> List[float]:
    """Convert decimal returns to percentage values."""
    return [r * 100 for r in decimal_values]


def extract_final_returns(
    portfolio_performances: Dict[str, Dict[int, float]]
) -> List[float]:
    """Extract final returns from all portfolio performance periods."""
    all_final_returns = []

    for _, performance in portfolio_performances.items():
        if not performance or len(performance) < 2:
            continue

        timestamps = sorted(performance.keys())
        final_return = performance[timestamps[-1]]
        all_final_returns.append(final_return)

    return all_final_returns


def calculate_percentiles(returns_array: np.ndarray) -> Dict[str, float]:
    """Calculate percentile statistics for returns."""
    percentiles = [25, 50, 75]
    return {
        f"percentile_{p}": float(np.percentile(returns_array, p) * 100)
        for p in percentiles
    }


def calculate_performance_statistics(
    all_final_returns: List[float],
    all_fee_returns: List[float] = None,
    all_il_returns: List[float] = None,
) -> Dict[str, float]:
    """Calculate performance statistics."""
    if not all_final_returns:
        return {}

    returns_array = np.array(all_final_returns)
    returns_pct = returns_array * 100

    stats = {
        "total_returns": {
            "mean_return_pct": float(np.mean(returns_pct)),
            "median_return_pct": float(np.median(returns_pct)),
            "std_return_pct": float(np.std(returns_pct)),
            "positive_periods": int(np.sum(returns_array > 0)),
            "total_periods": len(returns_array),
        }
    }

    # Add percentiles for total returns
    stats["total_returns"].update(calculate_percentiles(returns_array))

    # Add fee component statistics
    if all_fee_returns:
        fee_array = np.array(all_fee_returns)
        fee_pct = fee_array * 100
        stats["fee_returns"] = {
            "mean_return_pct": float(np.mean(fee_pct)),
            "median_return_pct": float(np.median(fee_pct)),
        }
        stats["fee_returns"].update(calculate_percentiles(fee_array))

    # Add impermanent loss component statistics
    if all_il_returns:
        il_array = np.array(all_il_returns)
        il_pct = il_array * 100
        stats["il_returns"] = {
            "mean_return_pct": float(np.mean(il_pct)),
            "median_return_pct": float(np.median(il_pct)),
        }
        stats["il_returns"].update(calculate_percentiles(il_array))

    return stats


def main() -> None:
    """Main execution function."""
    print("=== DeFi LP Portfolio Simulation ===\n")

    # 1. Get top EVM chains
    top_chains = get_top_evm_chains()

    # 2. Fetch historical data
    print("\nFetching historical TVL data per chain...")
    chain_tvls = {}
    for chain in top_chains:
        print(f"Fetching data for {chain}...")
        tvl_series = fetch_chain_history(chain)
        if tvl_series:
            # Apply forward-fill to handle missing data points
            filled_series = forward_fill_tvl_data(tvl_series)
            chain_tvls[chain] = filled_series

    print(f"Successfully fetched data for {len(chain_tvls)} chains")

    # 3. Fetch global TVL history
    global_tvl = fetch_global_tvl_history()
    if not global_tvl:
        print(
            "Warning: Failed to fetch global TVL data, calculations may be inaccurate"
        )

    # 4. Calculate TVL shares using global TVL
    start_date = datetime.now() - timedelta(days=ANALYSIS_MONTHS * 30)
    tvl_shares = calculate_tvl_shares(chain_tvls, global_tvl, start_date)

    # 5. Run portfolio simulation
    portfolio_performances, fee_performances, il_performances = (
        run_portfolio_simulation(list(chain_tvls.keys()), tvl_shares, start_date)
    )

    # 6. Extract final returns for each component
    final_returns = extract_final_returns(portfolio_performances)
    final_fee_returns = extract_final_returns(fee_performances)
    final_il_returns = extract_final_returns(il_performances)

    # 8. Calculate and save statistics
    stats = calculate_performance_statistics(
        final_returns, final_fee_returns, final_il_returns
    )

    print("\n=== PERFORMANCE STATISTICS ===")

    # Total returns
    total_stats = stats.get("total_returns", {})
    print("\nTOTAL RETURNS:")
    print(f"Mean: {total_stats.get('mean_return_pct', 0):.2f}%")
    print(f"25th Percentile: {total_stats.get('percentile_25', 0):.2f}%")
    print(f"50th Percentile (Median): {total_stats.get('percentile_50', 0):.2f}%")
    print(f"75th Percentile: {total_stats.get('percentile_75', 0):.2f}%")
    print(
        f"Positive Periods: {total_stats.get('positive_periods', 0)}/{total_stats.get('total_periods', 0)}"
    )

    # Fee component
    if "fee_returns" in stats:
        fee_stats = stats["fee_returns"]
        print("\nFEE COMPONENT:")
        print(f"Mean: {fee_stats.get('mean_return_pct', 0):.2f}%")
        print(f"25th Percentile: {fee_stats.get('percentile_25', 0):.2f}%")
        print(f"50th Percentile (Median): {fee_stats.get('percentile_50', 0):.2f}%")
        print(f"75th Percentile: {fee_stats.get('percentile_75', 0):.2f}%")

    # Impermanent Loss component
    if "il_returns" in stats:
        il_stats = stats["il_returns"]
        print("\nIMPERMANENT LOSS COMPONENT:")
        print(f"Mean: {il_stats.get('mean_return_pct', 0):.2f}%")
        print(f"25th Percentile: {il_stats.get('percentile_25', 0):.2f}%")
        print(f"50th Percentile (Median): {il_stats.get('percentile_50', 0):.2f}%")
        print(f"75th Percentile: {il_stats.get('percentile_75', 0):.2f}%")

    # Save results
    output_dir = Path("portfolio_results")
    output_dir.mkdir(exist_ok=True)

    results = {
        "configuration": {
            "chains": TOP_EVM_CHAINS,
            "analysis_months": ANALYSIS_MONTHS,
            "simulation_period_days": SIMULATION_PERIOD_DAYS,
            "daily_fee_rate": DAILY_FEE_RATE,
            "withdrawal_enabled": WITHDRAWAL_ENABLED,
            "withdrawal_timing_pct": WITHDRAWAL_TIMING_PCT,
            "withdrawal_amount_pct": WITHDRAWAL_AMOUNT_PCT,
        },
        "chains_analyzed": list(chain_tvls.keys()),
        "statistics": stats,
        "final_returns": final_returns,
        "final_fee_returns": final_fee_returns,
        "final_il_returns": final_il_returns,
    }

    with open(output_dir / "simulation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to portfolio_results/")
    print("- simulation_results.json: Detailed statistics and data")
    print("- portfolio_performance_overlay.png: Performance visualization")


if __name__ == "__main__":
    main()
