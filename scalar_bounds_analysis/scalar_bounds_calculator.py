#!/usr/bin/env python3
"""
Scalar Bounds Calculator for Prediction Markets

Enhanced TVL bounds calculation using:
- Log-returns instead of arithmetic returns
- Empirical quantiles instead of normal distribution
- Bootstrap confidence intervals
- Regime-aware volatility
- Cross-sectional priors for young series

Based on recommendations from scalar-bounds-design.md
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from scipy import stats
from dataclasses import dataclass, asdict

from lp_simulation_utils import cached_api_fetch, forward_fill_tvl_data


@dataclass
class ScalarBoundsResult:
    """Results from enhanced scalar bounds calculation."""

    chain: str
    current_tvl: float

    # Primary bounds (empirical quantiles)
    lower_bound: float
    upper_bound: float
    confidence_level: float

    # Alternative bounds for comparison
    bootstrap_lower: float
    bootstrap_upper: float
    regime_aware_lower: float
    regime_aware_upper: float

    # Statistics
    log_returns_mean: float
    log_returns_std: float
    volatility_30day: float
    volatility_annual: float
    num_periods: int
    data_days: int

    # Young series adjustments
    used_priors: bool
    prior_weight: float
    growth_cap_applied: bool
    age_cap_multiplier: float


class ScalarBoundsCalculator:
    """
    Enhanced bounds calculator using log-returns and empirical methods.
    """

    def __init__(self, chain: str, lookback_months: int = 12, period_days: int = 30):
        self.chain = chain
        self.lookback_months = lookback_months
        self.period_days = period_days
        self.tvl_data: Dict[int, float] = {}
        self.mature_chains = ["Base", "Arbitrum", "Ethereum", "Polygon"]

    def fetch_tvl_history(self) -> None:
        """Fetch historical TVL data for the chain."""
        print(f"Fetching TVL history for {self.chain}...")

        url = f"https://api.llama.fi/v2/historicalChainTvl/{self.chain}"
        cache_file = Path("../cache") / f"chain_tvl_{self.chain}.json"

        data = cached_api_fetch(url, cache_file)
        if not data:
            raise ValueError(f"Could not fetch TVL data for {self.chain}")

        # Convert to timestamp -> TVL mapping
        tvl_series = {}
        for entry in data:
            timestamp = entry.get("date")
            tvl = entry.get("tvl", 0)
            if timestamp and tvl > 0:
                tvl_series[timestamp] = tvl

        self.tvl_data = forward_fill_tvl_data(tvl_series)
        print(f"Fetched {len(self.tvl_data)} TVL data points")

    def calculate_log_returns(self, period_days: int = None) -> np.ndarray:
        """
        Calculate rolling log-returns for the specified period.
        Log-returns handle multiplicative growth better than arithmetic returns.
        """
        if period_days is None:
            period_days = self.period_days

        if not self.tvl_data:
            self.fetch_tvl_history()

        timestamps = sorted(self.tvl_data.keys())

        # Filter to lookback period
        end_date = datetime.fromtimestamp(timestamps[-1])
        start_date = end_date - timedelta(days=self.lookback_months * 30)
        start_timestamp = int(start_date.timestamp())

        filtered_timestamps = [t for t in timestamps if t >= start_timestamp]

        # Calculate log-returns
        log_returns = []
        period_seconds = period_days * 86400

        for t in filtered_timestamps:
            target_timestamp = t + period_seconds
            future_timestamps = [
                ts for ts in filtered_timestamps if ts >= target_timestamp
            ]

            if not future_timestamps:
                continue

            future_timestamp = future_timestamps[0]
            start_tvl = self.tvl_data[t]
            end_tvl = self.tvl_data[future_timestamp]

            if start_tvl > 0 and end_tvl > 0:
                # Log-return: log(end/start)
                log_return = np.log(end_tvl / start_tvl)
                log_returns.append(log_return)

        return np.array(log_returns)

    def empirical_quantile_bands(
        self, log_returns: np.ndarray, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate bounds using empirical quantiles from historical log-returns.
        This avoids normal distribution assumptions.
        """
        if len(log_returns) < 10:
            raise ValueError(
                f"Insufficient data ({len(log_returns)} periods) for quantile estimation"
            )

        # Calculate quantiles
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        q_low = np.quantile(log_returns, lower_quantile)
        q_high = np.quantile(log_returns, upper_quantile)

        # Convert log-returns to multiplicative factors
        current_tvl = self.tvl_data[max(self.tvl_data.keys())]
        lower_bound = current_tvl * np.exp(q_low)
        upper_bound = current_tvl * np.exp(q_high)

        return lower_bound, upper_bound

    def bootstrap_confidence_bands(
        self,
        log_returns: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        block_size: int = 5,
    ) -> Tuple[float, float]:
        """
        Use block bootstrap to compute confidence bands.
        Block bootstrap preserves autocorrelation structure.
        """
        if len(log_returns) < block_size * 2:
            # Fall back to simple bootstrap if not enough data for blocks
            block_size = 1

        n_returns = len(log_returns)
        bootstrap_quantiles = []

        for _ in range(n_bootstrap):
            # Block bootstrap: sample blocks of consecutive returns
            n_blocks = n_returns // block_size
            block_indices = np.random.choice(
                n_returns - block_size + 1, n_blocks, replace=True
            )

            bootstrap_sample = []
            for idx in block_indices:
                bootstrap_sample.extend(log_returns[idx : idx + block_size])

            # Trim to original length
            bootstrap_sample = bootstrap_sample[:n_returns]

            # Calculate quantiles for this bootstrap sample
            alpha = 1 - confidence_level
            q_low = np.quantile(bootstrap_sample, alpha / 2)
            q_high = np.quantile(bootstrap_sample, 1 - alpha / 2)
            bootstrap_quantiles.append((q_low, q_high))

        # Average the bootstrap quantiles
        avg_q_low = np.mean([q[0] for q in bootstrap_quantiles])
        avg_q_high = np.mean([q[1] for q in bootstrap_quantiles])

        current_tvl = self.tvl_data[max(self.tvl_data.keys())]
        lower_bound = current_tvl * np.exp(avg_q_low)
        upper_bound = current_tvl * np.exp(avg_q_high)

        return lower_bound, upper_bound

    def regime_aware_bounds(
        self,
        log_returns: np.ndarray,
        confidence_level: float = 0.95,
        vol_threshold_percentile: float = 0.75,
    ) -> Tuple[float, float]:
        """
        Calculate bounds with regime awareness (calm vs stress periods).
        Split data by volatility threshold and weight accordingly.
        """
        if len(log_returns) < 20:
            # Not enough data for regime detection
            return self.empirical_quantile_bands(log_returns, confidence_level)

        # Calculate rolling volatility (using 10-period windows)
        window = min(10, len(log_returns) // 4)
        rolling_std = np.array(
            [
                np.std(log_returns[max(0, i - window) : i + 1])
                for i in range(len(log_returns))
            ]
        )

        # Determine volatility threshold
        vol_threshold = np.percentile(rolling_std, vol_threshold_percentile * 100)

        # Split returns by regime
        calm_mask = rolling_std <= vol_threshold
        stress_mask = ~calm_mask

        calm_returns = log_returns[calm_mask] if np.any(calm_mask) else log_returns
        stress_returns = (
            log_returns[stress_mask] if np.any(stress_mask) else log_returns
        )

        # Calculate quantiles for each regime
        alpha = 1 - confidence_level

        if len(calm_returns) > 5:
            calm_q_low = np.quantile(calm_returns, alpha / 2)
            calm_q_high = np.quantile(calm_returns, 1 - alpha / 2)
        else:
            calm_q_low, calm_q_high = np.quantile(
                log_returns, [alpha / 2, 1 - alpha / 2]
            )

        if len(stress_returns) > 5:
            stress_q_low = np.quantile(stress_returns, alpha / 2)
            stress_q_high = np.quantile(stress_returns, 1 - alpha / 2)
        else:
            stress_q_low, stress_q_high = calm_q_low, calm_q_high

        # Weight by current volatility regime
        recent_vol = (
            np.std(log_returns[-window:])
            if len(log_returns) > window
            else np.std(log_returns)
        )
        stress_weight = (
            min(1.0, (recent_vol - vol_threshold) / vol_threshold)
            if recent_vol > vol_threshold
            else 0.0
        )
        calm_weight = 1.0 - stress_weight

        # Blend quantiles
        q_low = calm_weight * calm_q_low + stress_weight * stress_q_low
        q_high = calm_weight * calm_q_high + stress_weight * stress_q_high

        current_tvl = self.tvl_data[max(self.tvl_data.keys())]
        lower_bound = current_tvl * np.exp(q_low)
        upper_bound = current_tvl * np.exp(q_high)

        return lower_bound, upper_bound

    def get_cross_chain_priors(self) -> Dict[str, float]:
        """
        Extract volatility and growth statistics from mature chains.
        Used as priors for young series.
        """
        priors = {
            "log_returns_mean": [],
            "log_returns_std": [],
            "q_low": [],
            "q_high": [],
        }

        for mature_chain in self.mature_chains:
            if mature_chain == self.chain:
                continue

            try:
                # Try to load cached data
                cache_file = Path("../cache") / f"chain_tvl_{mature_chain}.json"
                if not cache_file.exists():
                    continue

                # Create temporary calculator for mature chain
                temp_calc = ScalarBoundsCalculator(mature_chain, self.lookback_months)
                temp_calc.fetch_tvl_history()

                # Get log returns - use same period as parent calculator
                log_returns = temp_calc.calculate_log_returns(self.period_days)
                if len(log_returns) < 30:
                    continue

                # Extract statistics
                priors["log_returns_mean"].append(np.mean(log_returns))
                priors["log_returns_std"].append(np.std(log_returns))
                priors["q_low"].append(np.quantile(log_returns, 0.025))
                priors["q_high"].append(np.quantile(log_returns, 0.975))

            except Exception as e:
                print(f"Could not get priors from {mature_chain}: {e}")
                continue

        # Calculate median of priors
        if priors["log_returns_mean"]:
            return {
                "log_returns_mean": np.median(priors["log_returns_mean"]),
                "log_returns_std": np.median(priors["log_returns_std"]),
                "q_low": np.median(priors["q_low"]),
                "q_high": np.median(priors["q_high"]),
            }
        else:
            # Default conservative priors if no mature chains available
            return {
                "log_returns_mean": 0.0,
                "log_returns_std": 0.15,  # ~15% monthly vol
                "q_low": -0.4,  # ~-33% lower bound
                "q_high": 0.5,  # ~+65% upper bound
            }

    def apply_young_series_adjustments(
        self, lower_bound: float, upper_bound: float, log_returns: np.ndarray
    ) -> Tuple[float, float, Dict]:
        """
        Apply adjustments for young series including:
        - Cross-sectional priors
        - Age-based caps
        - Growth-rate caps
        """
        timestamps = sorted(self.tvl_data.keys())
        data_days = (timestamps[-1] - timestamps[0]) // 86400
        current_tvl = self.tvl_data[timestamps[-1]]

        adjustments = {
            "used_priors": False,
            "prior_weight": 0.0,
            "growth_cap_applied": False,
            "age_cap_multiplier": None,
        }

        # Determine if this is a young series
        if data_days < 365:  # Less than 1 year of data

            # 1. Apply cross-sectional priors
            if data_days < 180:  # Less than 6 months
                priors = self.get_cross_chain_priors()

                # Weight priors based on data availability
                # More weight to priors when less data available
                prior_weight = max(0.0, 1.0 - (data_days / 180))
                adjustments["prior_weight"] = prior_weight
                adjustments["used_priors"] = True

                # Blend local statistics with priors
                if len(log_returns) > 0:
                    local_mean = np.mean(log_returns)
                    local_std = np.std(log_returns)
                    local_q_low = np.quantile(log_returns, 0.025)
                    local_q_high = np.quantile(log_returns, 0.975)

                    # Weighted average
                    blended_q_low = (
                        1 - prior_weight
                    ) * local_q_low + prior_weight * priors["q_low"]
                    blended_q_high = (
                        1 - prior_weight
                    ) * local_q_high + prior_weight * priors["q_high"]

                    # Recalculate bounds with blended quantiles
                    lower_bound = current_tvl * np.exp(blended_q_low)
                    upper_bound = current_tvl * np.exp(blended_q_high)

            # 2. Apply age-based caps
            if data_days < 90:
                age_cap_multiplier = 5.0
            elif data_days < 180:
                age_cap_multiplier = 3.0
            else:
                age_cap_multiplier = 2.0

            adjustments["age_cap_multiplier"] = age_cap_multiplier

            # 3. Check recent growth rate and apply additional cap if needed
            if len(log_returns) >= 7:
                # Calculate recent weekly growth
                week_ago_idx = max(0, len(timestamps) - 7)
                week_ago_tvl = self.tvl_data[timestamps[week_ago_idx]]
                weekly_growth = (
                    (current_tvl / week_ago_tvl - 1) if week_ago_tvl > 0 else 0
                )

                if weekly_growth > 0.5:  # More than 50% weekly growth
                    age_cap_multiplier = min(age_cap_multiplier, 2.5)
                    adjustments["growth_cap_applied"] = True

            # Apply caps
            upper_bound = min(upper_bound, current_tvl * age_cap_multiplier)
            lower_bound = max(lower_bound, current_tvl * 0.2)  # Floor at 20% of current

        return lower_bound, upper_bound, adjustments

    def calculate_bounds(self, confidence_level: float = 0.95) -> ScalarBoundsResult:
        """
        Calculate comprehensive bounds using multiple methods.
        """
        if not self.tvl_data:
            self.fetch_tvl_history()

        # Get log returns
        log_returns = self.calculate_log_returns()

        if len(log_returns) < 10:
            raise ValueError(
                f"Insufficient data ({len(log_returns)} periods) for bounds calculation"
            )

        # Calculate bounds using different methods
        empirical_lower, empirical_upper = self.empirical_quantile_bands(
            log_returns, confidence_level
        )
        bootstrap_lower, bootstrap_upper = self.bootstrap_confidence_bands(
            log_returns, confidence_level
        )
        regime_lower, regime_upper = self.regime_aware_bounds(
            log_returns, confidence_level
        )

        # Apply young series adjustments to empirical bounds (primary method)
        final_lower, final_upper, adjustments = self.apply_young_series_adjustments(
            empirical_lower, empirical_upper, log_returns
        )

        # Calculate statistics
        timestamps = sorted(self.tvl_data.keys())
        current_tvl = self.tvl_data[timestamps[-1]]
        data_days = (timestamps[-1] - timestamps[0]) // 86400

        log_returns_mean = np.mean(log_returns)
        log_returns_std = np.std(log_returns)

        # Convert log volatility to regular volatility approximation
        volatility_period = np.sqrt(np.exp(log_returns_std**2) - 1)
        periods_per_year = 365 / self.period_days
        volatility_annual = volatility_period * np.sqrt(periods_per_year)

        return ScalarBoundsResult(
            chain=self.chain,
            current_tvl=current_tvl,
            lower_bound=final_lower,
            upper_bound=final_upper,
            confidence_level=confidence_level,
            bootstrap_lower=bootstrap_lower,
            bootstrap_upper=bootstrap_upper,
            regime_aware_lower=regime_lower,
            regime_aware_upper=regime_upper,
            log_returns_mean=log_returns_mean,
            log_returns_std=log_returns_std,
            volatility_30day=volatility_period,
            volatility_annual=volatility_annual,
            num_periods=len(log_returns),
            data_days=data_days,
            used_priors=adjustments["used_priors"],
            prior_weight=adjustments["prior_weight"],
            growth_cap_applied=adjustments["growth_cap_applied"],
            age_cap_multiplier=adjustments["age_cap_multiplier"] or 0.0,
        )

    def visualize_comparison(
        self, result: ScalarBoundsResult, save_path: str = None
    ) -> None:
        """Visualize bounds comparison across different methods."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Get historical data
        timestamps = sorted(self.tvl_data.keys())
        tvls = [self.tvl_data[t] for t in timestamps]
        dates = [datetime.fromtimestamp(t) for t in timestamps]

        # Filter to recent data
        end_date = dates[-1]
        start_date = end_date - timedelta(days=self.lookback_months * 30)
        filtered_data = [(d, tvl) for d, tvl in zip(dates, tvls) if d >= start_date]
        filtered_dates = [d for d, _ in filtered_data]
        filtered_tvls = [tvl for _, tvl in filtered_data]

        # Plot 1: TVL history with all bounds
        ax1 = axes[0, 0]
        ax1.plot(
            filtered_dates, filtered_tvls, "b-", label=f"{self.chain} TVL", linewidth=2
        )
        ax1.axhline(
            result.current_tvl, color="green", linestyle="-", alpha=0.5, label="Current"
        )

        # Different bounds methods
        ax1.axhline(result.lower_bound, color="red", linestyle="-", linewidth=2)
        ax1.axhline(
            result.upper_bound,
            color="red",
            linestyle="-",
            linewidth=2,
            label="Final (Empirical + Adjustments)",
        )

        ax1.axhline(result.bootstrap_lower, color="orange", linestyle="--", alpha=0.5)
        ax1.axhline(
            result.bootstrap_upper,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label="Bootstrap",
        )

        ax1.axhline(result.regime_aware_lower, color="purple", linestyle=":", alpha=0.5)
        ax1.axhline(
            result.regime_aware_upper,
            color="purple",
            linestyle=":",
            alpha=0.5,
            label="Regime-Aware",
        )

        ax1.set_xlabel("Date")
        ax1.set_ylabel("TVL (USD)")
        ax1.set_title("TVL with Different Bound Methods")
        ax1.set_yscale("linear")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Log-returns distribution
        ax2 = axes[0, 1]
        log_returns = self.calculate_log_returns()
        ax2.hist(log_returns, bins=30, alpha=0.7, color="blue", edgecolor="black")
        ax2.axvline(
            np.mean(log_returns),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(log_returns):.3f}",
        )
        ax2.axvline(
            np.quantile(log_returns, 0.025),
            color="orange",
            linestyle="--",
            label="2.5% quantile",
        )
        ax2.axvline(
            np.quantile(log_returns, 0.975),
            color="orange",
            linestyle="--",
            label="97.5% quantile",
        )
        ax2.set_xlabel("30-day Log Return")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Log-Returns Distribution (n={len(log_returns)})")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Bounds comparison table
        ax3 = axes[1, 0]
        ax3.axis("tight")
        ax3.axis("off")

        table_data = [
            ["Method", "Lower Bound", "Upper Bound", "Width"],
            [
                "Empirical + Adj",
                f"${result.lower_bound/1e6:.1f}M",
                f"${result.upper_bound/1e6:.1f}M",
                f"{(result.upper_bound/result.lower_bound):.1f}x",
            ],
            [
                "Bootstrap",
                f"${result.bootstrap_lower/1e6:.1f}M",
                f"${result.bootstrap_upper/1e6:.1f}M",
                f"{(result.bootstrap_upper/result.bootstrap_lower):.1f}x",
            ],
            [
                "Regime-Aware",
                f"${result.regime_aware_lower/1e6:.1f}M",
                f"${result.regime_aware_upper/1e6:.1f}M",
                f"{(result.regime_aware_upper/result.regime_aware_lower):.1f}x",
            ],
        ]

        table = ax3.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Plot 4: Adjustments info
        ax4 = axes[1, 1]
        ax4.axis("off")

        info_text = f"""
Chain: {result.chain}
Current TVL: ${result.current_tvl/1e6:.1f}M
Data Days: {result.data_days}
Confidence Level: {result.confidence_level:.0%}

Statistics:
- Log Returns Mean: {result.log_returns_mean:.3f}
- Log Returns Std: {result.log_returns_std:.3f}
- 30-day Volatility: {result.volatility_30day:.1%}
- Annual Volatility: {result.volatility_annual:.1%}

Young Series Adjustments:
- Used Priors: {result.used_priors}
- Prior Weight: {result.prior_weight:.1%}
- Growth Cap Applied: {result.growth_cap_applied}
- Age Cap Multiplier: {result.age_cap_multiplier or 'N/A'}
        """

        ax4.text(
            0.1,
            0.5,
            info_text,
            fontsize=9,
            verticalalignment="center",
            family="monospace",
        )

        plt.suptitle(f"Scalar Bounds Analysis - {self.chain}", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

    def print_summary(self, result: ScalarBoundsResult) -> None:
        """Print comprehensive summary of bounds calculation."""
        print("\n" + "=" * 60)
        print(f"SCALAR BOUNDS ANALYSIS FOR {result.chain.upper()}")
        print("=" * 60)

        print(
            f"\nData: {result.data_days} days, {result.num_periods} {self.period_days}-day periods"
        )
        print(f"Current TVL: ${result.current_tvl/1e9:.2f}B")

        print(f"\nLog-Returns Statistics ({self.period_days}-day):")
        print(
            f"  Mean: {result.log_returns_mean:.4f} ({(np.exp(result.log_returns_mean)-1)*100:.2f}% equivalent)"
        )
        print(f"  Std Dev: {result.log_returns_std:.4f}")
        print(
            f"  {self.period_days}-day Volatility: {result.volatility_30day*100:.1f}%"
        )
        print(f"  Annualized Volatility: {result.volatility_annual*100:.1f}%")

        print(f"\nFINAL BOUNDS ({result.confidence_level:.0%} confidence):")
        print(
            f"  Lower: ${result.lower_bound/1e9:.2f}B ({(result.lower_bound/result.current_tvl-1)*100:+.1f}%)"
        )
        print(
            f"  Upper: ${result.upper_bound/1e9:.2f}B ({(result.upper_bound/result.current_tvl-1)*100:+.1f}%)"
        )
        print(f"  Ratio: {result.upper_bound/result.lower_bound:.1f}x")

        print(f"\nAlternative Methods:")
        print(
            f"  Bootstrap:    [${result.bootstrap_lower/1e9:.2f}B, ${result.bootstrap_upper/1e9:.2f}B]"
        )
        print(
            f"  Regime-Aware: [${result.regime_aware_lower/1e9:.2f}B, ${result.regime_aware_upper/1e9:.2f}B]"
        )

        if result.used_priors or result.age_cap_multiplier:
            print(f"\nYoung Series Adjustments Applied:")
            if result.used_priors:
                print(f"  - Cross-chain priors (weight: {result.prior_weight:.1%})")
            if result.age_cap_multiplier:
                print(f"  - Age cap: {result.age_cap_multiplier}x")
            if result.growth_cap_applied:
                print(f"  - Growth rate cap applied")

        print("=" * 60)


def main(chain: str = "Base", period_days: int = 30, lookback_months: int = 12):
    """Main execution function."""

    # Initialize calculator
    calculator = ScalarBoundsCalculator(
        chain=chain, lookback_months=lookback_months, period_days=period_days
    )

    # Calculate bounds
    result = calculator.calculate_bounds(confidence_level=0.95)

    # Print summary
    calculator.print_summary(result)

    # Save results  
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Save numerical results
    results_file = results_dir / f"{chain.lower()}_scalar_bounds_{period_days}d.json"
    with open(results_file, "w") as f:
        result_dict = asdict(result)
        result_dict["analysis_date"] = datetime.now().isoformat()
        json.dump(result_dict, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Visualize
    viz_file = results_dir / f"{chain.lower()}_scalar_bounds_{period_days}d_viz.png"
    calculator.visualize_comparison(result, save_path=str(viz_file))

    return result


if __name__ == "__main__":
    import sys

    chain = sys.argv[1] if len(sys.argv) > 1 else "Base"
    period_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    lookback_months = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    result = main(chain, period_days, lookback_months)
