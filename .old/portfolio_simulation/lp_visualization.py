#!/usr/bin/env python3
"""
LP Portfolio Return Distribution Visualizations

Creates intuitive, LP-friendly visualizations of return distributions
from the portfolio simulation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for professional charts
plt.style.use("default")
sns.set_palette("husl")


class LPReturnVisualizer:
    """Creates intuitive visualizations of LP return distributions."""

    def __init__(
        self,
        results_file: str = "../portfolio_simulation/results/simulation_results.json",
    ):
        self.results_file = Path(results_file)
        self.data = self._load_results()
        self.raw_data = self.data["raw_returns"]
        self.colors = {
            "total": "#2E86AB",
            "fee": "#A23B72",
            "il": "#F18F01",
            "external": "#6A4C93",
            "positive": "#27AE60",
            "negative": "#E74C3C",
        }

    def _load_results(self) -> Dict:
        """Load simulation results from JSON file."""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        with open(self.results_file, "r") as f:
            return json.load(f)

    def create_dashboard(self, output_dir: str = "lp_returns") -> None:
        """Create all visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        self._create_return_distributions(output_path)
        self._create_per_chain_distributions(output_path)

        print(f"✓ Visualizations saved to {output_path}")

    def _create_return_distributions(self, output_path: Path) -> None:
        """Create return distribution histograms for the aggregated portfolio."""
        self._create_and_save_distribution_plot(
            self.raw_data,
            "Aggregated Portfolio LP Return Distribution",
            output_path / "return_distributions.png",
        )

    def _create_per_chain_distributions(self, output_path: Path) -> None:
        """Create return distribution histograms for each individual chain."""
        per_chain_data = self.data.get("per_chain_raw_returns")
        if not per_chain_data:
            print("Info: Per-chain return data not found. Skipping individual plots.")
            return

        for chain_name, raw_data in per_chain_data.items():
            title = f"{chain_name} LP Return Distribution"
            output_file = output_path / f"{chain_name.lower()}_return_distributions.png"
            self._create_and_save_distribution_plot(raw_data, title, output_file)

    def _create_and_save_distribution_plot(
        self, raw_data: Dict, title: str, output_file: Path
    ) -> None:
        """Creates and saves a return distribution plot."""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Use raw data if available, otherwise fall back to generated distributions
        total_data = raw_data.get("total", [])
        fee_data = raw_data.get("fee", [])
        il_data = raw_data.get("il", [])
        external_data = raw_data.get("external", [])

        self._plot_histogram(ax1, total_data, "Total Returns", self.colors["total"])
        self._plot_histogram(ax2, fee_data, "Fee Returns", self.colors["fee"])
        self._plot_histogram(
            ax3, il_data, "Impermanent Loss Returns (Pool Only)", self.colors["il"]
        )
        self._plot_histogram(
            ax4,
            external_data,
            "External Token Returns (Outside AMM)",
            self.colors["external"],
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_histogram(self, ax, data: List[float], title: str, color: str) -> None:
        """Plot return distribution histogram."""
        # Ensure data is in the correct format
        if not data:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{title} (N=0)", fontsize=12, fontweight="bold")
            return

        # Convert to numpy array and handle any invalid values
        data_array = np.array(data)
        data_array = data_array[np.isfinite(data_array)]  # Remove any NaN or inf values
        num_samples = len(data_array)

        if num_samples == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{title} (N=0)", fontsize=12, fontweight="bold")
            return

        data_pct = data_array

        # Determine decimal precision based on data magnitude
        max_abs_val = np.max(np.abs(data_pct))
        if max_abs_val < 0.1:
            # For very small values (<0.1%), use 3 decimal places
            precision = 3
        elif max_abs_val < 1.0:
            # For small values (<1%), use 2 decimal places
            precision = 2
        else:
            # For larger values, use 1 decimal place
            precision = 1

        ax.hist(
            data_pct, bins=30, color=color, alpha=0.7, edgecolor="black", linewidth=0.5
        )

        mean_val = np.mean(data_pct)
        median_val = np.median(data_pct)

        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.{precision}f}%",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.{precision}f}%",
        )

        # Removed std dev, min, and max annotations per request

        ax.set_xlabel("Return (%)", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title(f"{title} (N={num_samples})", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)


def main():
    """Create all visualizations."""
    try:
        visualizer = LPReturnVisualizer()
        visualizer.create_dashboard()
        print("✓ All visualizations created successfully!")
        print("📊 Check the 'lp_returns' directory for the generated charts.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Make sure you have run the simulation first to generate results data.")


if __name__ == "__main__":
    main()
