#!/usr/bin/env python3
"""
Compare old (normal distribution) vs new (empirical/log-returns) bounds methods.
"""

import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


def load_results(chain: str) -> Dict:
    """Load results from different calculation methods."""
    results_dir = Path("portfolio_results")

    results = {}

    # Old method (normal distribution)
    old_file = results_dir / f"{chain.lower()}_tvl_bounds.json"
    if old_file.exists():
        with open(old_file) as f:
            results["old"] = json.load(f)

    # New scalar method (empirical/log-returns)
    new_file = results_dir / f"{chain.lower()}_scalar_bounds.json"
    if new_file.exists():
        with open(new_file) as f:
            results["new"] = json.load(f)

    # Young chains method (if applicable)
    young_file = results_dir / f"{chain.lower()}_tvl_bounds_young.json"
    if young_file.exists():
        with open(young_file) as f:
            results["young"] = json.load(f)

    return results


def compare_chains():
    """Compare bounds across chains and methods."""
    chains = ["Base", "Arbitrum", "Unichain"]

    # Create comparison table
    print("\n" + "=" * 100)
    print("BOUNDS COMPARISON: OLD (NORMAL) vs NEW (EMPIRICAL LOG-RETURNS)")
    print("=" * 100)

    for chain in chains:
        results = load_results(chain)

        print(f"\n{chain.upper()}")
        print("-" * 50)

        if "old" in results:
            old = results["old"]
            current = old["current_tvl"] / 1e9
            lower = old["lower_bound"] / 1e9
            upper = old["upper_bound"] / 1e9
            vol = old["volatility_30day"] * 100

            print(f"Old Method (Arithmetic + Normal):")
            print(f"  Current: ${current:.2f}B")
            print(f"  Bounds:  [${lower:.2f}B, ${upper:.2f}B]")
            print(
                f"  Range:   [{(lower/current-1)*100:+.1f}%, {(upper/current-1)*100:+.1f}%]"
            )
            print(f"  30d Vol: {vol:.1f}%")

        if "new" in results:
            new = results["new"]
            current = new["current_tvl"] / 1e9
            lower = new["lower_bound"] / 1e9
            upper = new["upper_bound"] / 1e9
            vol = new["volatility_30day"] * 100

            print(f"\nNew Method (Log-Returns + Empirical):")
            print(f"  Current: ${current:.2f}B")
            print(f"  Bounds:  [${lower:.2f}B, ${upper:.2f}B]")
            print(
                f"  Range:   [{(lower/current-1)*100:+.1f}%, {(upper/current-1)*100:+.1f}%]"
            )
            print(f"  30d Vol: {vol:.1f}%")

            # Show alternative methods
            print(
                f"  Bootstrap:    [${new['bootstrap_lower']/1e9:.2f}B, ${new['bootstrap_upper']/1e9:.2f}B]"
            )
            print(
                f"  Regime-Aware: [${new['regime_aware_lower']/1e9:.2f}B, ${new['regime_aware_upper']/1e9:.2f}B]"
            )

            # Show adjustments if applied
            if new.get("used_priors"):
                print(f"  Prior Weight: {new['prior_weight']*100:.1f}%")
            if new.get("age_cap_multiplier"):
                print(f"  Age Cap: {new['age_cap_multiplier']}x")

        if "young" in results and chain == "Unichain":
            young = results["young"]
            rec_lower = young["bounds"]["recommended"]["lower"] / 1e9
            rec_upper = young["bounds"]["recommended"]["upper"] / 1e9

            print(f"\nYoung Series Method (Alternative):")
            print(f"  Recommended: [${rec_lower:.2f}B, ${rec_upper:.2f}B]")

    print("\n" + "=" * 100)
    print("KEY IMPROVEMENTS IN NEW METHOD:")
    print("=" * 100)
    print(
        """
1. LOG-RETURNS: Better handling of multiplicative growth
   - Old: (M_t+30 - M_t) / M_t
   - New: log(M_t+30 / M_t)

2. EMPIRICAL QUANTILES: No normal distribution assumption
   - Old: mean ± z*std (assumes normality)
   - New: actual 0.5% and 99.5% quantiles from data

3. BOOTSTRAP: Accounts for autocorrelation
   - Block bootstrap preserves time series structure

4. REGIME AWARENESS: Adapts to market conditions
   - Separates calm vs stress periods
   - Weights by current volatility regime

5. YOUNG SERIES: Smart handling of limited data
   - Cross-chain priors for stability
   - Age-based caps prevent unrealistic bounds
   - Growth-rate caps for explosive series
"""
    )

    # Create visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for i, chain in enumerate(chains):
        ax = axes[i]
        results = load_results(chain)

        if "old" in results and "new" in results:
            old = results["old"]
            new = results["new"]

            current = old["current_tvl"] / 1e9

            # Plot bounds
            methods = ["Old\n(Normal)", "New\n(Empirical)", "Bootstrap", "Regime"]
            lowers = [
                old["lower_bound"] / 1e9,
                new["lower_bound"] / 1e9,
                new["bootstrap_lower"] / 1e9,
                new["regime_aware_lower"] / 1e9,
            ]
            uppers = [
                old["upper_bound"] / 1e9,
                new["upper_bound"] / 1e9,
                new["bootstrap_upper"] / 1e9,
                new["regime_aware_upper"] / 1e9,
            ]

            x_pos = np.arange(len(methods))

            # Plot error bars
            for j, (method, lower, upper) in enumerate(zip(methods, lowers, uppers)):
                ax.plot([j, j], [lower, upper], "o-", linewidth=2, markersize=8)
                ax.text(j, upper + 0.1, f"${upper:.1f}B", ha="center", fontsize=8)
                ax.text(j, lower - 0.1, f"${lower:.1f}B", ha="center", fontsize=8)

            # Current value line
            ax.axhline(
                current,
                color="green",
                linestyle="--",
                alpha=0.5,
                label=f"Current: ${current:.1f}B",
            )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods)
            ax.set_ylabel("TVL (Billions USD)")
            ax.set_title(f"{chain}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Bounds Comparison: Old vs New Methods", fontsize=14)
    plt.tight_layout()

    output_path = Path("portfolio_results") / "bounds_methods_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison visualization to {output_path}")


if __name__ == "__main__":
    compare_chains()
