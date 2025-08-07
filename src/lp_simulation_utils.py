#!/usr/bin/env python3
"""
Shared utilities for LP Portfolio Simulations

This module contains shared logic for simulating UP/DOWN token liquidity pools
across different TVL data sources (total DeFi TVL or individual chains).
"""

from __future__ import annotations

import json
import requests
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration parameters for the LP portfolio simulation."""

    analysis_months: int = 24
    simulation_period_days: int = 30
    period_spacing_days: int = 1
    fee_rate: float = 0.003
    withdrawal_enabled: bool = True
    withdrawal_timing_pct: float = 0.25
    withdrawal_amount_pct: float = 0.7


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
        """Calculate UP and DOWN token prices based on TVL performance ratio.

        UP now redeems at 1.5x starting TVL (instead of 2x).
        When tvl_ratio = 1.5, up_price = 1.0 (full redemption).
        """
        # UP reaches full redemption (1.0) when TVL reaches 1.5x
        # Formula: up_price = (2/3) * tvl_ratio, capped between 0.01 and 0.99
        up_price = min(0.99, max(0.01, (2 / 3) * tvl_ratio))
        return up_price, 1.0 - up_price

    def calculate_initial_amm_deposits(
        self, total_tokens_per_type: float
    ) -> Tuple[float, float, float, float]:
        """Calculate initial AMM deposits and external balances.

        Args:
            total_tokens_per_type: Total number of UP and DOWN tokens to mint (e.g., 1000 each)

        Returns:
            Tuple of (up_tokens_in_amm, down_tokens_in_amm, external_up_tokens, external_down_tokens)
        """
        # At start, tvl_ratio = 1.0, so prices are:
        up_price, down_price = self.calculate_token_prices(1.0)

        # We mint equal amounts of UP and DOWN (e.g., 1000 each)
        # Some goes to AMM with equal USD value, rest stays external
        # Let's say we put V USD value of each token in the AMM
        # That means: up_in_amm = V / up_price, down_in_amm = V / down_price

        # We want to maximize AMM usage while keeping external balances >= 0
        # The constraint is that we can't put more tokens than we have
        max_up_value = total_tokens_per_type * up_price  # Max USD value from UP tokens
        max_down_value = (
            total_tokens_per_type * down_price
        )  # Max USD value from DOWN tokens

        # The AMM gets equal USD value from each, limited by the smaller pool
        amm_value_per_token = min(max_up_value, max_down_value)

        up_tokens_in_amm = amm_value_per_token / up_price
        down_tokens_in_amm = amm_value_per_token / down_price

        # Calculate external balances (what's left after AMM deposit)
        external_up_tokens = total_tokens_per_type - up_tokens_in_amm
        external_down_tokens = total_tokens_per_type - down_tokens_in_amm

        return (
            up_tokens_in_amm,
            down_tokens_in_amm,
            external_up_tokens,
            external_down_tokens,
        )

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
        tvl_data: Dict[int, float],
        start_timestamp: int,
        end_timestamp: int,
    ) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]], float]:
        """Simulate UP/DOWN token LP pool over one period.

        Args:
            tvl_data: Dictionary of {timestamp: tvl_value}
            start_timestamp: Period start timestamp
            end_timestamp: Period end timestamp

        Returns:
            Tuple of (pool_values, external_holdings, accumulated_fees)
        """
        if not tvl_data:
            return {}, {}, 0.0

        available_times = [
            t for t in tvl_data.keys() if start_timestamp <= t <= end_timestamp
        ]
        available_times.sort()

        if len(available_times) < 2:
            return {}, {}, 0.0

        start_tvl = tvl_data[available_times[0]]
        if start_tvl <= 0:
            return {}, {}, 0.0

        return self._simulate_pool_internal(
            tvl_data, available_times, start_tvl, start_timestamp, end_timestamp
        )

    def _simulate_pool_internal(
        self,
        tvl_data: Dict[int, float],
        available_times: List[int],
        start_tvl: float,
        start_timestamp: int,
        end_timestamp: int,
    ) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]], float]:
        """Internal pool simulation logic with reduced nesting."""
        # Calculate initial setup
        _, _, external_up_tokens, external_down_tokens = (
            self.calculate_initial_amm_deposits(1000.0)
        )
        k = 1000.0 * 1000.0  # This will be adjusted based on actual initial deposits

        # Recalculate k based on actual initial AMM deposits
        up_in_amm, down_in_amm, external_up_tokens, external_down_tokens = (
            self.calculate_initial_amm_deposits(1000.0)
        )
        k = up_in_amm * down_in_amm

        withdrawal_executed = False
        accumulated_fees = 0.0
        last_timestamp = available_times[0]
        last_up_price, last_down_price = None, None

        period_duration = end_timestamp - start_timestamp
        withdrawal_timestamp = start_timestamp + (
            period_duration * self.config.withdrawal_timing_pct
        )

        pool_values = {}
        external_holdings = {}

        for timestamp in available_times:
            current_tvl = tvl_data[timestamp]
            tvl_ratio = current_tvl / start_tvl
            up_price, down_price = self.calculate_token_prices(tvl_ratio)

            accumulated_fees += self._calculate_period_fees(
                k,
                up_price,
                down_price,
                timestamp,
                last_timestamp,
                last_up_price,
                last_down_price,
            )
            last_timestamp = timestamp
            last_up_price = up_price
            last_down_price = down_price

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
        last_up_price: float = None,
        last_down_price: float = None,
    ) -> float:
        """Calculate fees for the time period based on implied trade volume."""
        days_elapsed = (timestamp - last_timestamp) / 86400
        if days_elapsed <= 0 or last_up_price is None or last_down_price is None:
            return 0.0

        # Calculate token amounts at current prices
        up_tokens = np.sqrt(k * down_price / up_price)
        down_tokens = np.sqrt(k * up_price / down_price)

        # Calculate what token amounts would have been at last prices with same k
        last_up_tokens = np.sqrt(k * last_down_price / last_up_price)
        last_down_tokens = np.sqrt(k * last_up_price / last_down_price)

        # Calculate the implied trade volume (absolute change in token amounts)
        up_volume = abs(up_tokens - last_up_tokens) * up_price
        down_volume = abs(down_tokens - last_down_tokens) * down_price

        # Total trade volume is the average of both sides (since trades affect both tokens)
        total_trade_volume = (up_volume + down_volume) / 2

        # Apply fee rate to the trade volume
        return total_trade_volume * self.config.fee_rate

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

    def calculate_returns(
        self,
        pool_values: Dict[int, float],
        external_holdings: Dict[int, Tuple[float, float]],
        tvl_data: Dict[int, float],
        start_tvl: float,
        timestamps: List[int],
        fees: float,
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """Calculate total, fee, and IL returns over time."""
        total_returns = {}
        fee_returns = {}
        il_returns = {}

        for t in timestamps:
            if t in pool_values and t in tvl_data:
                current_tvl = tvl_data[t]
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

                total_returns[t] = total_return
                fee_returns[t] = fee_return
                il_returns[t] = il_return

        return total_returns, fee_returns, il_returns

    def simulate_single_period(
        self,
        tvl_data: Dict[int, float],
        start_timestamp: int,
        end_timestamp: int,
        debug: bool = False,
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[str, float]]:
        """Simulate portfolio for a single period."""
        pool_values, external_holdings, fees = self.pool_simulator.simulate_pool_period(
            tvl_data, start_timestamp, end_timestamp
        )

        if not pool_values:
            return {}, {}, {}, {}

        timestamps = sorted(pool_values.keys())
        if len(timestamps) < 2 or pool_values[timestamps[0]] <= 0:
            return {}, {}, {}, {}

        start_tvl = tvl_data[timestamps[0]]
        total_returns, fee_returns, il_returns = self.calculate_returns(
            pool_values, external_holdings, tvl_data, start_tvl, timestamps, fees
        )

        debug_info = self._calculate_debug_info(
            tvl_data, timestamps, il_returns, start_tvl
        )

        if debug:
            self._print_period_debug(
                pool_values, external_holdings, tvl_data, fees, timestamps, debug_info
            )

        return total_returns, fee_returns, il_returns, debug_info

    def _calculate_debug_info(
        self,
        tvl_data: Dict[int, float],
        timestamps: List[int],
        il_returns: Dict[int, float],
        start_tvl: float,
    ) -> Dict[str, float]:
        """Calculate debug information for the period."""
        if not timestamps:
            return {}

        end_tvl = tvl_data[timestamps[-1]]
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
            "up_down_ratio_change_pct": price_ratio_change * 100,
            "start_up_price": start_up_price,
            "end_up_price": end_up_price,
            "start_price_ratio": start_price_ratio,
            "end_price_ratio": end_price_ratio,
            "start_tvl": start_tvl,
            "end_tvl": end_tvl,
            "tvl_change_pct": (end_tvl - start_tvl) / start_tvl * 100,
            "il_return_pct": (
                il_returns[timestamps[-1]] * 100 if timestamps[-1] in il_returns else 0
            ),
        }

    def _print_period_debug(
        self, pool_values, external_holdings, tvl_data, fees, timestamps, debug_info
    ):
        """Print debug information for the period."""
        start_tvl = tvl_data[timestamps[0]]
        final_value = (
            pool_values[timestamps[-1]]
            + fees
            + sum(
                ext * price
                for ext, price in zip(
                    external_holdings[timestamps[-1]],
                    self.pool_simulator.calculate_token_prices(
                        tvl_data[timestamps[-1]] / start_tvl
                    ),
                )
            )
        )
        print(
            f"  Total: total={final_value:.2f}, return={((final_value/1000-1)*100):.2f}%"
        )
        print(f"  TVL change: {debug_info['tvl_change_pct']:.2f}%")
        print(f"  UP:DOWN ratio change: {debug_info['up_down_ratio_change_pct']:.2f}%")

    def run_portfolio_simulation(
        self,
        tvl_data_source: Union[Dict[int, float], Dict[str, Dict[int, float]]],
        start_date: datetime,
        data_type: str = "single",  # "single" for total TVL, "multi" for per-chain
    ) -> Tuple[
        Dict[str, Dict[int, float]],
        Dict[str, Dict[int, float]],
        Dict[str, Dict[int, float]],
        Dict[str, Union[Dict[str, float], Dict[str, Dict[str, float]]]],
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

            if data_type == "single":
                # Single TVL data source (like total DeFi TVL)
                total_returns, fee_returns, il_returns, debug_info = (
                    self.simulate_single_period(
                        tvl_data_source,
                        start_timestamp,
                        end_timestamp,
                        debug=(period_count < 3),
                    )
                )
            else:
                # Multi-chain data source - this would need to be implemented
                # by the calling code for chain-specific logic
                raise NotImplementedError(
                    "Multi-chain simulation not implemented in base class"
                )

            portfolio_performances[period_key] = total_returns
            fee_performances[period_key] = fee_returns
            il_performances[period_key] = il_returns
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
        percentiles = [10, 25, 50, 75]
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

    def print_performance_statistics(self, stats: Dict) -> None:
        """Print performance statistics."""
        print("\n=== PERFORMANCE STATISTICS ===")

        total_stats = stats.get("total_returns", {})
        print("\nTOTAL RETURNS:")
        print(f"Mean: {total_stats.get('mean_return_pct', 0):.2f}%")
        print(f"10th Percentile (Worst): {total_stats.get('percentile_10', 0):.2f}%")
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
            print(f"10th Percentile (Worst): {fee_stats.get('percentile_10', 0):.2f}%")
            print(f"25th Percentile: {fee_stats.get('percentile_25', 0):.2f}%")
            print(f"50th Percentile (Median): {fee_stats.get('percentile_50', 0):.2f}%")
            print(f"75th Percentile: {fee_stats.get('percentile_75', 0):.2f}%")

        if "il_returns" in stats:
            il_stats = stats["il_returns"]
            print("\nIMPERMANENT LOSS COMPONENT:")
            print(f"Mean: {il_stats.get('mean_return_pct', 0):.2f}%")
            print(f"10th Percentile (Worst): {il_stats.get('percentile_10', 0):.2f}%")
            print(f"25th Percentile: {il_stats.get('percentile_25', 0):.2f}%")
            print(f"50th Percentile (Median): {il_stats.get('percentile_50', 0):.2f}%")
            print(f"75th Percentile: {il_stats.get('percentile_75', 0):.2f}%")


def find_latest_timestamp(timestamps: List[int], target: int) -> int:
    """Find the latest timestamp <= target from a list of timestamps."""
    available = [t for t in timestamps if t <= target]
    return max(available) if available else None


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
