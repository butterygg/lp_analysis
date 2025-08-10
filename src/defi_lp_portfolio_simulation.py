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
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

from lp_simulation_utils import (
    SimulationConfig,
    cached_api_fetch,
    LPPoolSimulator,
    PortfolioAnalyzer as BasePortfolioAnalyzer,
    forward_fill_tvl_data,
    find_latest_timestamp,
)


def create_chain_specific_config() -> SimulationConfig:
    """Create a SimulationConfig with chain-specific TVL ratio configurations."""
    config = SimulationConfig()

    # Override the default chain_tvl_ratios with chain-specific values
    config.chain_tvl_ratios = {
        "Arbitrum": {
            "min_tvl_ratio": 0.82,
            "max_tvl_ratio": 1.3,
            "min_up_price": 0.01,
            "max_up_price": 1.0,
        },
        "Base": {
            "min_tvl_ratio": 0.82,
            "max_tvl_ratio": 1.4,
            "min_up_price": 0.01,
            "max_up_price": 1.0,
        },
        "Unichain": {
            "min_tvl_ratio": 0.75,
            "max_tvl_ratio": 1.66,
            "min_up_price": 0.01,
            "max_up_price": 1.0,
        },
    }

    # Set top EVM chains
    config.top_evm_chains = [
        "Arbitrum",
        "Base",
        "Unichain",
        # "BSC",
        # "Avalanche",
    ]

    return config


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
        return forward_fill_tvl_data(tvl_series, max_gap_days)

    def _find_latest_timestamp(self, timestamps: List[int], target: int) -> int:
        """Find the latest timestamp <= target from a list of timestamps."""
        return find_latest_timestamp(timestamps, target)


def get_top_evm_chains() -> List[str]:
    """Return the hardcoded list of top EVM chains."""
    config = SimulationConfig()
    print(
        f"Using top {len(config.top_evm_chains)} EVM chains: {', '.join(config.top_evm_chains)}"
    )
    return config.top_evm_chains


class ChainPortfolioAnalyzer(BasePortfolioAnalyzer):
    """Specialized portfolio analyzer for multi-chain simulations."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

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
        # Convert chain TVL data to format expected by base class
        chain_tvl_dict = tvl_data.chain_tvls[chain]
        start_tvl = chain_tvl_dict[timestamps[0]]

        return self.calculate_returns(
            pool_values,
            external_holdings,
            chain_tvl_dict,
            start_tvl,
            timestamps,
            fees,
            chain,
        )

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
            # Get chain data and simulate using base class
            if chain not in tvl_data.chain_tvls:
                continue

            chain_tvl_dict = tvl_data.chain_tvls[chain]
            pool_values, external_holdings, fees = (
                self.pool_simulator.simulate_pool_period(
                    chain_tvl_dict, start_timestamp, end_timestamp, chain
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

        chain_tvl_dict = tvl_data.chain_tvls[chain]
        start_tvl = chain_tvl_dict[timestamps[0]]
        debug_info = self._calculate_debug_info_base(
            chain_tvl_dict, timestamps, il_contributions, start_tvl, chain
        )

        return {chain: debug_info}

    def _calculate_debug_info_base(
        self,
        tvl_data: Dict[int, float],
        timestamps: List[int],
        il_returns: Dict[int, float],
        start_tvl: float,
        chain: str = "default",
    ) -> Dict[str, float]:
        """Calculate debug information using base class logic."""
        return super()._calculate_debug_info(
            tvl_data, timestamps, il_returns, start_tvl, chain
        )

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
        start_tvl = tvl_data.chain_tvls[chain][timestamps[0]]
        final_value = (
            pool_values[timestamps[-1]]
            + fees
            + sum(
                ext * price
                for ext, price in zip(
                    external_holdings[timestamps[-1]],
                    self.pool_simulator.calculate_token_prices(
                        tvl_data.chain_tvls[chain][timestamps[-1]] / start_tvl, chain
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


class SimulationWorkflow:
    """Orchestrates the complete simulation workflow."""

    def __init__(self):
        self.config = create_chain_specific_config()
        self.tvl_data = ChainTVLData(self.config)
        self.analyzer = ChainPortfolioAnalyzer(self.config)

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
        self.analyzer.print_performance_statistics(stats)

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
                "fee_rate": self.config.fee_rate,
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
