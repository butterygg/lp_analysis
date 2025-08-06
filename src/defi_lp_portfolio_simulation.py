#!/usr/bin/env python3
"""
LP Portfolio Simulation Script

This script:
1. Fetches top N DeFi chains from DeFiLlama
2. Uses historical absolute TVL of each chain over 24 months
3. Simulates UP/DOWN token liquidity pool performance
4. Analyzes portfolio performance based on absolute TVL changes
"""

from __future__ import annotations

import json
import requests
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration parameters for the LP portfolio simulation."""

    top_evm_chains: List[str] = None
    analysis_months: int = 24
    simulation_period_days: int = 30
    period_spacing_days: int = 1
    daily_fee_rate: float = 0.003
    withdrawal_enabled: bool = True
    withdrawal_timing_pct: float = 0.25
    withdrawal_amount_pct: float = 0.7

    def __post_init__(self):
        if self.top_evm_chains is None:
            self.top_evm_chains = [
                "Ethereum",
                "BSC",
                "Base",
                "Arbitrum",
                "Avalanche",
            ]


class ChainTVLData:
    """Manages TVL data for multiple chains with caching and preprocessing."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.chain_tvls: Dict[str, Dict[int, float]] = {}

    def load_chain_data(self) -> None:
        """Load TVL data for all configured chains."""
        print(
            f"Using top {len(self.config.top_evm_chains)} EVM chains: {', '.join(self.config.top_evm_chains)}"
        )
        print("Fetching historical TVL data per chain...")

        for chain in self.config.top_evm_chains:
            print(f"Fetching data for {chain}...")
            tvl_series = self._fetch_chain_history(chain)
            if tvl_series:
                filled_series = self._forward_fill_tvl_data(tvl_series)
                self.chain_tvls[chain] = filled_series

        print(f"Successfully fetched data for {len(self.chain_tvls)} chains")

    def prepare_for_analysis(self, start_date: datetime) -> None:
        """Filter TVL data to the analysis period."""
        print("Preparing chain TVL data for analysis period...")

        start_timestamp = int(start_date.timestamp())
        filtered_tvls = {}

        for chain, tvl_data in self.chain_tvls.items():
            filtered_data = {
                t: tvl for t, tvl in tvl_data.items() if t >= start_timestamp
            }
            if filtered_data:
                filtered_tvls[chain] = filtered_data
                print(f"{chain}: {len(filtered_data)} data points")

        self.chain_tvls = filtered_tvls

    def get_tvl_at_timestamp(self, chain: str, timestamp: int) -> float:
        """Get TVL value for a chain at or before the given timestamp."""
        if chain not in self.chain_tvls:
            return 0.0

        tvl_data = self.chain_tvls[chain]
        latest_time = self._find_latest_timestamp(list(tvl_data.keys()), timestamp)
        return tvl_data[latest_time] if latest_time else 0.0

    def get_available_chains(self) -> List[str]:
        """Get list of chains with available data."""
        return list(self.chain_tvls.keys())

    def _fetch_chain_history(self, chain: str) -> Dict[int, float]:
        """Fetch historical TVL data for a chain."""
        cache_file = Path("cache") / f"chain_{chain}_history.json"
        url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
        data = cached_api_fetch(url, cache_file)

        if not data:
            return {}

        return {int(entry["date"]): float(entry["tvl"]) for entry in data}

    def _forward_fill_tvl_data(
        self, tvl_series: Dict[int, float], max_gap_days: int = 7
    ) -> Dict[int, float]:
        """Forward-fill missing TVL data points to handle short outages."""
        if not tvl_series:
            return tvl_series

        timestamps = sorted(tvl_series.keys())
        if len(timestamps) < 2:
            return tvl_series

        filled_series = tvl_series.copy()

        for i in range(len(timestamps) - 1):
            current_time = timestamps[i]
            next_time = timestamps[i + 1]
            gap_days = (next_time - current_time) / 86400

            if gap_days <= max_gap_days and gap_days > 1:
                current_value = tvl_series[current_time]
                fill_time = current_time + 86400
                while fill_time < next_time:
                    existing_timestamp = None
                    for existing_time in tvl_series.keys():
                        if abs(existing_time - fill_time) <= 3600:
                            existing_timestamp = existing_time
                            break

                    if not existing_timestamp:
                        filled_series[fill_time] = current_value
                    fill_time += 86400

        return filled_series

    def _find_latest_timestamp(self, timestamps: List[int], target: int) -> int:
        """Find the latest timestamp <= target from a list of timestamps."""
        available = [t for t in timestamps if t <= target]
        return max(available) if available else None


def get_top_evm_chains() -> List[str]:
    """Return the hardcoded list of top EVM chains."""
    config = SimulationConfig()
    print(
        f"Using top {len(config.top_evm_chains)} EVM chains: {', '.join(config.top_evm_chains)}"
    )
    return config.top_evm_chains


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


class LPPoolSimulator:
    """Simulates UP/DOWN token liquidity pool operations."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def calculate_token_prices(self, tvl_ratio: float) -> Tuple[float, float]:
        """Calculate UP and DOWN token prices based on TVL performance ratio."""
        up_price = min(0.99, max(0.01, 0.5 * tvl_ratio))
        return up_price, 1.0 - up_price

    def execute_withdrawal(
        self, k: float, up_price: float, down_price: float
    ) -> Tuple[float, float, float]:
        """Execute proportional withdrawal from LP pool."""
        pre_withdrawal_up = np.sqrt(k * down_price / up_price)
        pre_withdrawal_down = np.sqrt(k * up_price / down_price)

        withdrawn_up = pre_withdrawal_up * self.config.withdrawal_amount_pct
        withdrawn_down = pre_withdrawal_down * self.config.withdrawal_amount_pct

        remaining_factor = 1.0 - self.config.withdrawal_amount_pct
        new_k = k * (remaining_factor**2)

        return withdrawn_up, withdrawn_down, new_k

    def calculate_pool_value(
        self, k: float, up_price: float, down_price: float
    ) -> float:
        """Calculate total pool value using constant product formula."""
        up_tokens = np.sqrt(k * down_price / up_price)
        down_tokens = np.sqrt(k * up_price / down_price)
        return up_tokens * up_price + down_tokens * down_price

    def simulate_pool_period(
        self,
        chain: str,
        tvl_data: ChainTVLData,
        start_timestamp: int,
        end_timestamp: int,
    ) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]], float]:
        """Simulate UP/DOWN token LP pool for one chain over one period."""
        if chain not in tvl_data.chain_tvls:
            return {}, {}, 0.0

        chain_data = tvl_data.chain_tvls[chain]
        available_times = [
            t for t in chain_data.keys() if start_timestamp <= t <= end_timestamp
        ]
        available_times.sort()

        if len(available_times) < 2:
            return {}, {}, 0.0

        start_tvl = chain_data[available_times[0]]
        if start_tvl <= 0:
            return {}, {}, 0.0

        return self._simulate_pool_internal(
            chain_data, available_times, start_tvl, start_timestamp, end_timestamp
        )

    def _simulate_pool_internal(
        self,
        chain_data: Dict[int, float],
        available_times: List[int],
        start_tvl: float,
        start_timestamp: int,
        end_timestamp: int,
    ) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]], float]:
        """Internal pool simulation logic with reduced nesting."""
        k = 1000.0 * 1000.0
        external_up_tokens, external_down_tokens = 0.0, 0.0
        withdrawal_executed = False
        accumulated_fees = 0.0
        last_timestamp = available_times[0]

        period_duration = end_timestamp - start_timestamp
        withdrawal_timestamp = start_timestamp + (
            period_duration * self.config.withdrawal_timing_pct
        )

        pool_values = {}
        external_holdings = {}

        for timestamp in available_times:
            current_tvl = chain_data[timestamp]
            tvl_ratio = current_tvl / start_tvl
            up_price, down_price = self.calculate_token_prices(tvl_ratio)

            accumulated_fees += self._calculate_period_fees(
                k, up_price, down_price, timestamp, last_timestamp
            )
            last_timestamp = timestamp

            if self._should_execute_withdrawal(
                withdrawal_executed, timestamp, withdrawal_timestamp
            ):
                withdrawn_up, withdrawn_down, k = self.execute_withdrawal(
                    k, up_price, down_price
                )
                external_up_tokens += withdrawn_up
                external_down_tokens += withdrawn_down
                withdrawal_executed = True

            pool_values[timestamp] = self.calculate_pool_value(k, up_price, down_price)
            external_holdings[timestamp] = (external_up_tokens, external_down_tokens)

        return pool_values, external_holdings, accumulated_fees

    def _calculate_period_fees(
        self,
        k: float,
        up_price: float,
        down_price: float,
        timestamp: int,
        last_timestamp: int,
    ) -> float:
        """Calculate fees for the time period."""
        days_elapsed = (timestamp - last_timestamp) / 86400
        if days_elapsed <= 0:
            return 0.0

        current_pool_value = self.calculate_pool_value(k, up_price, down_price)
        return current_pool_value * self.config.daily_fee_rate * days_elapsed

    def _should_execute_withdrawal(
        self, withdrawal_executed: bool, timestamp: int, withdrawal_timestamp: int
    ) -> bool:
        """Check if withdrawal should be executed."""
        return (
            self.config.withdrawal_enabled
            and not withdrawal_executed
            and timestamp >= withdrawal_timestamp
        )


class PortfolioAnalyzer:
    """Handles portfolio analysis, statistics, and reporting."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.pool_simulator = LPPoolSimulator(config)

    def calculate_chain_contribution(
        self,
        chain: str,
        pool_values: Dict[int, float],
        external_holdings: Dict[int, Tuple[float, float]],
        tvl_data: ChainTVLData,
        timestamps: List[int],
        fees: float,
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """Calculate a chain's contribution to portfolio returns over time.
        Returns: (total_contributions, fee_contributions, il_contributions)
        """
        total_contributions = {}
        fee_contributions = {}
        il_contributions = {}
        start_tvl = tvl_data.chain_tvls[chain][timestamps[0]]

        for t in timestamps:
            if (
                t in pool_values
                and chain in tvl_data.chain_tvls
                and t in tvl_data.chain_tvls[chain]
            ):
                current_tvl = tvl_data.chain_tvls[chain][t]
                tvl_ratio = current_tvl / start_tvl
                up_price, down_price = self.pool_simulator.calculate_token_prices(
                    tvl_ratio
                )

                ext_up, ext_down = external_holdings[t]
                external_value = ext_up * up_price + ext_down * down_price

                total_value = pool_values[t] + external_value + fees
                total_return = total_value / 1000.0 - 1
                fee_return = fees / 1000.0
                portfolio_value_no_fees = pool_values[t] + external_value
                il_return = portfolio_value_no_fees / 1000.0 - 1

                total_contributions[t] = total_return
                fee_contributions[t] = fee_return
                il_contributions[t] = il_return

        return total_contributions, fee_contributions, il_contributions

    def simulate_single_period(
        self,
        chains: List[str],
        tvl_data: ChainTVLData,
        start_timestamp: int,
        end_timestamp: int,
        debug: bool = False,
    ) -> Tuple[
        Dict[int, float],
        Dict[int, float],
        Dict[int, float],
        Dict[str, Dict[str, float]],
    ]:
        """Simulate portfolio for a single period."""
        portfolio_value = {}
        fee_portfolio_value = {}
        il_portfolio_value = {}
        chains_with_data_at_timestamp = {}
        successful_chains = 0
        debug_info = {}

        for chain in chains:
            pool_values, external_holdings, fees = (
                self.pool_simulator.simulate_pool_period(
                    chain, tvl_data, start_timestamp, end_timestamp
                )
            )

            if not pool_values:
                continue

            timestamps = sorted(pool_values.keys())
            if len(timestamps) < 2 or pool_values[timestamps[0]] <= 0:
                continue

            successful_chains += 1
            total_contributions, fee_contributions, il_contributions = (
                self.calculate_chain_contribution(
                    chain, pool_values, external_holdings, tvl_data, timestamps, fees
                )
            )

            debug_info.update(
                self._calculate_debug_info(
                    chain, tvl_data, timestamps, il_contributions
                )
            )
            self._add_to_portfolio(
                total_contributions,
                fee_contributions,
                il_contributions,
                portfolio_value,
                fee_portfolio_value,
                il_portfolio_value,
                chains_with_data_at_timestamp,
            )

            if debug and successful_chains <= 3:
                self._print_chain_debug(
                    chain, pool_values, external_holdings, tvl_data, fees, timestamps
                )

        self._normalize_portfolio_values(
            portfolio_value,
            fee_portfolio_value,
            il_portfolio_value,
            chains_with_data_at_timestamp,
        )

        if debug:
            self._print_period_summary(portfolio_value, successful_chains, debug_info)

        return portfolio_value, fee_portfolio_value, il_portfolio_value, debug_info

    def _calculate_debug_info(
        self,
        chain: str,
        tvl_data: ChainTVLData,
        timestamps: List[int],
        il_contributions: Dict[int, float],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate debug information for a chain."""
        if not timestamps:
            return {}

        start_tvl = tvl_data.chain_tvls[chain][timestamps[0]]
        end_tvl = tvl_data.chain_tvls[chain][timestamps[-1]]
        start_ratio = 1.0
        end_ratio = end_tvl / start_tvl
        start_up_price, start_down_price = self.pool_simulator.calculate_token_prices(
            start_ratio
        )
        end_up_price, end_down_price = self.pool_simulator.calculate_token_prices(
            end_ratio
        )

        start_price_ratio = start_up_price / start_down_price
        end_price_ratio = end_up_price / end_down_price
        price_ratio_change = (end_price_ratio - start_price_ratio) / start_price_ratio

        return {
            chain: {
                "up_down_ratio_change_pct": price_ratio_change * 100,
                "start_up_price": start_up_price,
                "end_up_price": end_up_price,
                "start_price_ratio": start_price_ratio,
                "end_price_ratio": end_price_ratio,
                "start_tvl": start_tvl,
                "end_tvl": end_tvl,
                "tvl_change_pct": (end_tvl - start_tvl) / start_tvl * 100,
                "il_return_pct": (
                    il_contributions[timestamps[-1]] * 100
                    if timestamps[-1] in il_contributions
                    else 0
                ),
            }
        }

    def _add_to_portfolio(
        self,
        total_contributions,
        fee_contributions,
        il_contributions,
        portfolio_value,
        fee_portfolio_value,
        il_portfolio_value,
        chains_with_data_at_timestamp,
    ):
        """Add chain contributions to portfolio totals."""
        for t in total_contributions:
            portfolio_value[t] = portfolio_value.get(t, 0) + total_contributions[t]
            fee_portfolio_value[t] = (
                fee_portfolio_value.get(t, 0) + fee_contributions[t]
            )
            il_portfolio_value[t] = il_portfolio_value.get(t, 0) + il_contributions[t]
            chains_with_data_at_timestamp[t] = (
                chains_with_data_at_timestamp.get(t, 0) + 1
            )

    def _normalize_portfolio_values(
        self,
        portfolio_value,
        fee_portfolio_value,
        il_portfolio_value,
        chains_with_data_at_timestamp,
    ):
        """Normalize portfolio values by number of contributing chains."""
        for t in portfolio_value:
            if chains_with_data_at_timestamp.get(t, 0) > 0:
                portfolio_value[t] /= chains_with_data_at_timestamp[t]
                fee_portfolio_value[t] /= chains_with_data_at_timestamp[t]
                il_portfolio_value[t] /= chains_with_data_at_timestamp[t]

    def _print_chain_debug(
        self, chain, pool_values, external_holdings, tvl_data, fees, timestamps
    ):
        """Print debug information for a chain."""
        final_value = (
            pool_values[timestamps[-1]]
            + fees
            + sum(
                ext * price
                for ext, price in zip(
                    external_holdings[timestamps[-1]],
                    self.pool_simulator.calculate_token_prices(
                        tvl_data.chain_tvls[chain][timestamps[-1]]
                        / tvl_data.chain_tvls[chain][timestamps[0]]
                    ),
                )
            )
        )
        print(
            f"  {chain}: total={final_value:.2f}, return={((final_value/1000-1)*100):.2f}%"
        )

    def _print_period_summary(self, portfolio_value, successful_chains, debug_info):
        """Print summary for the period."""
        if portfolio_value:
            final_return = portfolio_value[max(portfolio_value.keys())]
            print(
                f"  Period summary: {successful_chains} chains, portfolio return: {(final_return*100):.2f}%"
            )
            if debug_info:
                avg_ratio_change = np.mean(
                    [info["up_down_ratio_change_pct"] for info in debug_info.values()]
                )
                print(f"  Average UP:DOWN ratio change: {avg_ratio_change:.2f}%")
        else:
            print(f"  No successful chains in this period")

    def run_portfolio_simulation(
        self, chains: List[str], tvl_data: ChainTVLData, start_date: datetime
    ) -> Tuple[
        Dict[str, Dict[int, float]],
        Dict[str, Dict[int, float]],
        Dict[str, Dict[int, float]],
        Dict[str, Dict[str, Dict[str, float]]],
    ]:
        """Run portfolio simulation with overlapping 1-month periods."""
        print(
            f"Running overlapping portfolio simulations (periods spaced {self.config.period_spacing_days} days apart)..."
        )

        portfolio_performances = {}
        fee_performances = {}
        il_performances = {}
        all_debug_info = {}

        current_date = start_date
        end_analysis_date = start_date + timedelta(
            days=self.config.analysis_months * 30
        )
        max_start_date = end_analysis_date - timedelta(
            days=self.config.simulation_period_days
        )

        period_count = 0
        while current_date <= max_start_date:
            period_end = current_date + timedelta(
                days=self.config.simulation_period_days
            )
            period_key = (
                f"period_{period_count:03d}_{current_date.strftime('%Y-%m-%d')}"
            )

            if period_count % 30 == 0:
                print(f"Simulating period: {period_key}")

            start_timestamp = int(current_date.timestamp())
            end_timestamp = int(period_end.timestamp())

            portfolio_value, fee_value, il_value, debug_info = (
                self.simulate_single_period(
                    chains,
                    tvl_data,
                    start_timestamp,
                    end_timestamp,
                    debug=(period_count < 3),
                )
            )

            portfolio_performances[period_key] = portfolio_value
            fee_performances[period_key] = fee_value
            il_performances[period_key] = il_value
            all_debug_info[period_key] = debug_info

            current_date += timedelta(days=self.config.period_spacing_days)
            period_count += 1

        print(f"Generated {period_count} overlapping 1-month periods")
        return portfolio_performances, fee_performances, il_performances, all_debug_info

    @staticmethod
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

    @staticmethod
    def calculate_percentiles(returns_array: np.ndarray) -> Dict[str, float]:
        """Calculate percentile statistics for returns."""
        percentiles = [25, 50, 75]
        return {
            f"percentile_{p}": float(np.percentile(returns_array, p) * 100)
            for p in percentiles
        }

    def calculate_performance_statistics(
        self,
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

        stats["total_returns"].update(self.calculate_percentiles(returns_array))

        if all_fee_returns:
            fee_array = np.array(all_fee_returns)
            fee_pct = fee_array * 100
            stats["fee_returns"] = {
                "mean_return_pct": float(np.mean(fee_pct)),
                "median_return_pct": float(np.median(fee_pct)),
            }
            stats["fee_returns"].update(self.calculate_percentiles(fee_array))

        if all_il_returns:
            il_array = np.array(all_il_returns)
            il_pct = il_array * 100
            stats["il_returns"] = {
                "mean_return_pct": float(np.mean(il_pct)),
                "median_return_pct": float(np.median(il_pct)),
            }
            stats["il_returns"].update(self.calculate_percentiles(il_array))

        return stats


class SimulationWorkflow:
    """Orchestrates the complete simulation workflow."""

    def __init__(self):
        self.config = SimulationConfig()
        self.tvl_data = ChainTVLData(self.config)
        self.analyzer = PortfolioAnalyzer(self.config)

    def run_complete_simulation(self) -> Dict:
        """Run the complete simulation workflow."""
        print("=== DeFi LP Portfolio Simulation ===\n")

        # Load and prepare data
        self.tvl_data.load_chain_data()
        start_date = datetime.now() - timedelta(days=self.config.analysis_months * 30)
        self.tvl_data.prepare_for_analysis(start_date)

        # Run simulation
        chains = self.tvl_data.get_available_chains()
        portfolio_performances, fee_performances, il_performances, debug_info = (
            self.analyzer.run_portfolio_simulation(chains, self.tvl_data, start_date)
        )

        # Extract and analyze results
        final_returns = self.analyzer.extract_final_returns(portfolio_performances)
        final_fee_returns = self.analyzer.extract_final_returns(fee_performances)
        final_il_returns = self.analyzer.extract_final_returns(il_performances)

        stats = self.analyzer.calculate_performance_statistics(
            final_returns, final_fee_returns, final_il_returns
        )

        self._print_results(stats, debug_info)
        return self._save_results(
            stats, final_returns, final_fee_returns, final_il_returns, debug_info
        )

    def _print_results(self, stats: Dict, debug_info: Dict) -> None:
        """Print analysis results."""
        self._print_debug_analysis(debug_info)
        self._print_performance_statistics(stats)

    def _print_debug_analysis(self, debug_info: Dict) -> None:
        """Print debug analysis of price ratio changes."""
        print("\n=== DEBUGGING: UP:DOWN PRICE RATIO CHANGES VS IL ===")
        all_ratio_changes = []
        all_il_returns = []

        for _, period_debug in debug_info.items():
            for chain, chain_debug in period_debug.items():
                all_ratio_changes.append(chain_debug["up_down_ratio_change_pct"])
                all_il_returns.append(chain_debug["il_return_pct"])

        if all_ratio_changes:
            avg_ratio_change = np.mean(all_ratio_changes)
            std_ratio_change = np.std(all_ratio_changes)
            avg_il = np.mean(all_il_returns)
            std_il = np.std(all_il_returns)

            print(
                f"Average UP:DOWN ratio change: {avg_ratio_change:.2f}% (std: {std_ratio_change:.2f}%)"
            )
            print(f"Average IL return: {avg_il:.2f}% (std: {std_il:.2f}%)")

            # Show sample periods
            print("\nSample periods (first 10):")
            sample_periods = list(debug_info.keys())[:10]
            for period in sample_periods:
                period_debug = debug_info[period]
                if period_debug:
                    chain = list(period_debug.keys())[0]
                    chain_debug = period_debug[chain]
                    print(f"{period}, {chain}:")
                    start_tvl_b = chain_debug["start_tvl"] / 1e9
                    end_tvl_b = chain_debug["end_tvl"] / 1e9
                    print(
                        f"  TVL: ${start_tvl_b:.2f}B → ${end_tvl_b:.2f}B ({chain_debug['tvl_change_pct']:.2f}%)"
                    )
                    print(
                        f"  UP:DOWN ratio change: {chain_debug['up_down_ratio_change_pct']:.2f}%, IL: {chain_debug['il_return_pct']:.2f}%"
                    )

            print(
                "\nNote: 'IL' here represents portfolio value change (excluding fees)"
            )
            print(
                "Token prices are calculated based on absolute TVL changes rather than market share changes."
            )

    def _print_performance_statistics(self, stats: Dict) -> None:
        """Print performance statistics."""
        print("\n=== PERFORMANCE STATISTICS ===")

        total_stats = stats.get("total_returns", {})
        print("\nTOTAL RETURNS:")
        print(f"Mean: {total_stats.get('mean_return_pct', 0):.2f}%")
        print(f"25th Percentile: {total_stats.get('percentile_25', 0):.2f}%")
        print(f"50th Percentile (Median): {total_stats.get('percentile_50', 0):.2f}%")
        print(f"75th Percentile: {total_stats.get('percentile_75', 0):.2f}%")
        print(
            f"Positive Periods: {total_stats.get('positive_periods', 0)}/{total_stats.get('total_periods', 0)}"
        )

        if "fee_returns" in stats:
            fee_stats = stats["fee_returns"]
            print("\nFEE COMPONENT:")
            print(f"Mean: {fee_stats.get('mean_return_pct', 0):.2f}%")
            print(f"25th Percentile: {fee_stats.get('percentile_25', 0):.2f}%")
            print(f"50th Percentile (Median): {fee_stats.get('percentile_50', 0):.2f}%")
            print(f"75th Percentile: {fee_stats.get('percentile_75', 0):.2f}%")

        if "il_returns" in stats:
            il_stats = stats["il_returns"]
            print("\nIMPERMANENT LOSS COMPONENT:")
            print(f"Mean: {il_stats.get('mean_return_pct', 0):.2f}%")
            print(f"25th Percentile: {il_stats.get('percentile_25', 0):.2f}%")
            print(f"50th Percentile (Median): {il_stats.get('percentile_50', 0):.2f}%")
            print(f"75th Percentile: {il_stats.get('percentile_75', 0):.2f}%")

    def _save_results(
        self,
        stats: Dict,
        final_returns: List[float],
        final_fee_returns: List[float],
        final_il_returns: List[float],
        debug_info: Dict,
    ) -> Dict:
        """Save results to file."""
        output_dir = Path("portfolio_results")
        output_dir.mkdir(exist_ok=True)

        all_ratio_changes = []
        for _, period_debug in debug_info.items():
            for _, chain_debug in period_debug.items():
                all_ratio_changes.append(chain_debug["up_down_ratio_change_pct"])

        results = {
            "configuration": {
                "chains": self.config.top_evm_chains,
                "analysis_months": self.config.analysis_months,
                "simulation_period_days": self.config.simulation_period_days,
                "daily_fee_rate": self.config.daily_fee_rate,
                "withdrawal_enabled": self.config.withdrawal_enabled,
                "withdrawal_timing_pct": self.config.withdrawal_timing_pct,
                "withdrawal_amount_pct": self.config.withdrawal_amount_pct,
            },
            "chains_analyzed": self.tvl_data.get_available_chains(),
            "statistics": stats,
            "final_returns": final_returns,
            "final_fee_returns": final_fee_returns,
            "final_il_returns": final_il_returns,
            "debug_ratio_changes": all_ratio_changes[:100] if all_ratio_changes else [],
        }

        with open(output_dir / "simulation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to portfolio_results/")
        print("- simulation_results.json: Detailed statistics and data")

        return results


def main() -> None:
    """Main execution function."""
    workflow = SimulationWorkflow()
    workflow.run_complete_simulation()


if __name__ == "__main__":
    main()
