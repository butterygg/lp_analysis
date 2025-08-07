#!/usr/bin/env python3
"""
LP Portfolio Simulation Script - Total DeFi TVL

This script:
1. Fetches total DeFi TVL from DeFiLlama
2. Uses historical absolute TVL over 24 months
3. Simulates UP/DOWN token liquidity pool performance
4. Analyzes portfolio performance based on absolute TVL changes
"""

from __future__ import annotations

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from lp_simulation_utils import (
    SimulationConfig,
    cached_api_fetch,
    PortfolioAnalyzer,
    forward_fill_tvl_data,
    find_latest_timestamp,
)


class TotalTVLData:
    """Manages total DeFi TVL data with caching and preprocessing."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.total_tvl: Dict[int, float] = {}

    def load_total_tvl_data(self) -> None:
        """Load total DeFi TVL data."""
        print("Fetching total DeFi TVL data...")

        tvl_series = self._fetch_total_tvl_history()
        if tvl_series:
            filled_series = self._forward_fill_tvl_data(tvl_series)
            self.total_tvl = filled_series

        print(f"Successfully fetched {len(self.total_tvl)} data points")

    def prepare_for_analysis(self, start_date: datetime) -> None:
        """Filter TVL data to the analysis period."""
        print("Preparing total TVL data for analysis period...")

        start_timestamp = int(start_date.timestamp())
        filtered_data = {
            t: tvl for t, tvl in self.total_tvl.items() if t >= start_timestamp
        }

        if filtered_data:
            self.total_tvl = filtered_data
            print(f"Total DeFi TVL: {len(filtered_data)} data points")

    def get_tvl_at_timestamp(self, timestamp: int) -> float:
        """Get TVL value at or before the given timestamp."""
        if not self.total_tvl:
            return 0.0

        latest_time = self._find_latest_timestamp(
            list(self.total_tvl.keys()), timestamp
        )
        return self.total_tvl[latest_time] if latest_time else 0.0

    def _fetch_total_tvl_history(self) -> Dict[int, float]:
        """Fetch historical total DeFi TVL data."""
        cache_file = Path("cache") / "total_defi_tvl_history.json"
        url = "https://api.llama.fi/v2/historicalChainTvl"
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


class SimulationWorkflow:
    """Orchestrates the complete simulation workflow."""

    def __init__(self):
        self.config = SimulationConfig()
        self.tvl_data = TotalTVLData(self.config)
        self.analyzer = PortfolioAnalyzer(self.config)

    def run_complete_simulation(self) -> Dict:
        """Run the complete simulation workflow."""
        print("=== Total DeFi LP Portfolio Simulation ===\n")

        # Load and prepare data
        self.tvl_data.load_total_tvl_data()
        start_date = datetime.now() - timedelta(days=self.config.analysis_months * 30)
        self.tvl_data.prepare_for_analysis(start_date)

        # Run simulation
        portfolio_performances, fee_performances, il_performances, debug_info = (
            self.analyzer.run_portfolio_simulation(
                self.tvl_data.total_tvl, start_date, "single"
            )
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
        """Print debug analysis of price ratio changes vs IL."""
        print("\n=== DEBUGGING: UP:DOWN PRICE RATIO CHANGES VS IL ===")
        all_ratio_changes = []
        all_il_returns = []

        for _, period_debug in debug_info.items():
            if period_debug:  # Check if period_debug is not empty
                all_ratio_changes.append(period_debug["up_down_ratio_change_pct"])
                all_il_returns.append(period_debug["il_return_pct"])

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
                    print(f"{period}:")
                    start_tvl_b = period_debug["start_tvl"] / 1e9
                    end_tvl_b = period_debug["end_tvl"] / 1e9
                    print(
                        f"  TVL: ${start_tvl_b:.2f}B → ${end_tvl_b:.2f}B ({period_debug['tvl_change_pct']:.2f}%)"
                    )
                    print(
                        f"  UP:DOWN ratio change: {period_debug['up_down_ratio_change_pct']:.2f}%, IL: {period_debug['il_return_pct']:.2f}%"
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
            if period_debug:  # Check if period_debug is not empty
                all_ratio_changes.append(period_debug["up_down_ratio_change_pct"])

        results = {
            "configuration": {
                "analysis_months": self.config.analysis_months,
                "simulation_period_days": self.config.simulation_period_days,
                "fee_rate": self.config.fee_rate,
                "withdrawal_enabled": self.config.withdrawal_enabled,
                "withdrawal_timing_pct": self.config.withdrawal_timing_pct,
                "withdrawal_amount_pct": self.config.withdrawal_amount_pct,
            },
            "market_type": "total_defi_tvl",
            "statistics": stats,
            "final_returns": final_returns,
            "final_fee_returns": final_fee_returns,
            "final_il_returns": final_il_returns,
            "debug_ratio_changes": all_ratio_changes[:100] if all_ratio_changes else [],
        }

        with open(output_dir / "total_defi_simulation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to portfolio_results/")
        print("- total_defi_simulation_results.json: Detailed statistics and data")

        return results


def main() -> None:
    """Main execution function."""
    workflow = SimulationWorkflow()
    workflow.run_complete_simulation()


if __name__ == "__main__":
    main()
