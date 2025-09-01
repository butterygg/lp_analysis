#!/usr/bin/env python3
"""
Scalar Bounds Calculator with Recent Data Focus

Calculate bounds using only recent historical data (e.g., last 3 months).
Useful for adapting to regime changes and recent market conditions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict

from lp_simulation_utils import cached_api_fetch, forward_fill_tvl_data


@dataclass
class RecentBoundsResult:
    """Results from recent-data bounds calculation."""
    chain: str
    current_tvl: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    
    # Period info
    period_days: int
    recency_months: int
    
    # Statistics
    log_returns_mean: float
    log_returns_std: float
    volatility_period: float
    volatility_annual: float
    num_periods: int
    
    # Quantiles
    quantile_005: float
    quantile_025: float
    quantile_50: float
    quantile_975: float
    quantile_995: float
    
    # Extremes (max observed)
    max_drawdown: float
    max_increase: float
    max_bounds_lower: float
    max_bounds_upper: float
    
    # Date range
    start_date: str
    end_date: str


class RecentBoundsCalculator:
    """
    Calculate bounds using only recent data.
    """
    
    def __init__(self, chain: str, period_days: int = 21, recency_months: int = 3):
        self.chain = chain
        self.period_days = period_days
        self.recency_months = recency_months
        self.tvl_data: Dict[int, float] = {}
        
    def fetch_tvl_history(self) -> None:
        """Fetch historical TVL data for the chain."""
        print(f"Fetching TVL history for {self.chain}...")
        
        url = f"https://api.llama.fi/v2/historicalChainTvl/{self.chain}"
        cache_file = Path("cache") / f"chain_tvl_{self.chain}.json"
        
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
    
    def calculate_recent_log_returns(self) -> Tuple[np.ndarray, datetime, datetime]:
        """
        Calculate log-returns using only data from the last N months.
        Returns: (log_returns, start_date, end_date)
        """
        if not self.tvl_data:
            self.fetch_tvl_history()
            
        timestamps = sorted(self.tvl_data.keys())
        
        # Define the recency window
        end_timestamp = timestamps[-1]
        end_date = datetime.fromtimestamp(end_timestamp)
        
        # Start from N months ago
        start_date = end_date - timedelta(days=self.recency_months * 30)
        start_timestamp = int(start_date.timestamp())
        
        # Filter timestamps to recency window
        recent_timestamps = [t for t in timestamps if t >= start_timestamp]
        
        print(f"\nUsing data from {start_date.date()} to {end_date.date()}")
        print(f"Total days in window: {len(recent_timestamps)}")
        
        # Calculate log-returns for the specified period
        log_returns = []
        period_seconds = self.period_days * 86400
        
        for t in recent_timestamps:
            target_timestamp = t + period_seconds
            
            # Find the closest future timestamp
            future_timestamps = [ts for ts in recent_timestamps if ts >= target_timestamp]
            
            if not future_timestamps:
                continue
                
            future_timestamp = future_timestamps[0]
            
            # Only use if the future point is within reasonable distance
            actual_days = (future_timestamp - t) / 86400
            if actual_days > self.period_days * 1.5:  # Skip if gap is too large
                continue
                
            start_tvl = self.tvl_data[t]
            end_tvl = self.tvl_data[future_timestamp]
            
            if start_tvl > 0 and end_tvl > 0:
                # Log-return
                log_return = np.log(end_tvl / start_tvl)
                log_returns.append(log_return)
        
        print(f"Calculated {len(log_returns)} {self.period_days}-day returns from recent data")
        
        return np.array(log_returns), start_date, end_date
    
    def calculate_bounds(self, confidence_level: float = 0.99) -> RecentBoundsResult:
        """
        Calculate bounds using only recent data.
        """
        # Get recent log returns
        log_returns, start_date, end_date = self.calculate_recent_log_returns()
        
        if len(log_returns) < 5:
            raise ValueError(f"Insufficient recent data ({len(log_returns)} periods) for bounds calculation")
        
        # Calculate quantiles
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        q_005 = np.quantile(log_returns, 0.005)
        q_025 = np.quantile(log_returns, 0.025)
        q_50 = np.quantile(log_returns, 0.50)
        q_975 = np.quantile(log_returns, 0.975)
        q_995 = np.quantile(log_returns, 0.995)
        
        # Primary bounds (99% CI)
        q_low = np.quantile(log_returns, lower_quantile)
        q_high = np.quantile(log_returns, upper_quantile)
        
        # Get current TVL
        current_tvl = self.tvl_data[max(self.tvl_data.keys())]
        
        # Convert to bounds
        lower_bound = current_tvl * np.exp(q_low)
        upper_bound = current_tvl * np.exp(q_high)
        
        # Calculate max observed extremes
        max_drawdown_log = np.min(log_returns)
        max_increase_log = np.max(log_returns)
        
        # Convert to percentage and bounds
        max_drawdown = np.exp(max_drawdown_log) - 1  # As a ratio (negative)
        max_increase = np.exp(max_increase_log) - 1  # As a ratio (positive)
        
        # Max bounds based on observed extremes
        max_bounds_lower = current_tvl * np.exp(max_drawdown_log)
        max_bounds_upper = current_tvl * np.exp(max_increase_log)
        
        # Calculate statistics
        log_returns_mean = np.mean(log_returns)
        log_returns_std = np.std(log_returns)
        
        # Volatility calculations
        volatility_period = np.sqrt(np.exp(log_returns_std**2) - 1)
        periods_per_year = 365 / self.period_days
        volatility_annual = volatility_period * np.sqrt(periods_per_year)
        
        return RecentBoundsResult(
            chain=self.chain,
            current_tvl=current_tvl,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            period_days=self.period_days,
            recency_months=self.recency_months,
            log_returns_mean=log_returns_mean,
            log_returns_std=log_returns_std,
            volatility_period=volatility_period,
            volatility_annual=volatility_annual,
            num_periods=len(log_returns),
            quantile_005=q_005,
            quantile_025=q_025,
            quantile_50=q_50,
            quantile_975=q_975,
            quantile_995=q_995,
            max_drawdown=max_drawdown,
            max_increase=max_increase,
            max_bounds_lower=max_bounds_lower,
            max_bounds_upper=max_bounds_upper,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
    
    def visualize_analysis(self, result: RecentBoundsResult, save_path: str = None) -> None:
        """Visualize the recent data analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Get recent log returns for plotting
        log_returns, _, _ = self.calculate_recent_log_returns()
        
        # Plot 1: Distribution of log-returns
        ax1 = axes[0, 0]
        ax1.hist(log_returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
        
        # Add quantile lines
        ax1.axvline(result.quantile_005, color='red', linestyle='--', alpha=0.7, label='0.5% / 99.5%')
        ax1.axvline(result.quantile_995, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(result.quantile_025, color='orange', linestyle='--', alpha=0.5, label='2.5% / 97.5%')
        ax1.axvline(result.quantile_975, color='orange', linestyle='--', alpha=0.5)
        ax1.axvline(result.quantile_50, color='green', linestyle='-', alpha=0.7, label='Median')
        
        ax1.set_xlabel(f'{result.period_days}-day Log Return')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of Recent Returns (n={result.num_periods})')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: TVL history with focus on recent period
        ax2 = axes[0, 1]
        
        timestamps = sorted(self.tvl_data.keys())
        tvls = [self.tvl_data[t] for t in timestamps]
        dates = [datetime.fromtimestamp(t) for t in timestamps]
        
        # Plot full history in gray
        ax2.plot(dates, tvls, 'gray', alpha=0.3, label='Full History')
        
        # Highlight recent period
        recent_start = datetime.fromisoformat(result.start_date)
        recent_mask = [d >= recent_start for d in dates]
        recent_dates = [d for d, m in zip(dates, recent_mask) if m]
        recent_tvls = [tvl for tvl, m in zip(tvls, recent_mask) if m]
        
        ax2.plot(recent_dates, recent_tvls, 'b-', linewidth=2, label=f'Recent {result.recency_months} Months')
        
        # Add bounds
        ax2.axhline(result.current_tvl, color='green', linestyle='-', alpha=0.5, label='Current')
        ax2.axhline(result.lower_bound, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(result.upper_bound, color='red', linestyle='--', alpha=0.7, label='99% Bounds')
        ax2.axhline(result.max_bounds_lower, color='darkred', linestyle=':', alpha=0.5)
        ax2.axhline(result.max_bounds_upper, color='darkred', linestyle=':', alpha=0.5, label='Max Observed')
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('TVL (USD)')
        ax2.set_title(f'{self.chain} TVL - Recent Period Focus')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'${x/1e9:.2f}B' if x >= 1e9 else f'${x/1e6:.0f}M'
        ))
        
        # Plot 3: Quantile comparison
        ax3 = axes[1, 0]
        
        quantiles = [0.5, 2.5, 50, 97.5, 99.5]
        quantile_values = [
            result.quantile_005,
            result.quantile_025,
            result.quantile_50,
            result.quantile_975,
            result.quantile_995
        ]
        
        # Convert to percentage returns for readability
        pct_returns = [(np.exp(q) - 1) * 100 for q in quantile_values]
        
        colors = ['red', 'orange', 'green', 'orange', 'red']
        bars = ax3.bar(range(len(quantiles)), pct_returns, color=colors, alpha=0.6)
        
        # Add value labels
        for bar, val in zip(bars, pct_returns):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top')
        
        ax3.set_xticks(range(len(quantiles)))
        ax3.set_xticklabels([f'{q}%' for q in quantiles])
        ax3.set_ylabel(f'{result.period_days}-day Return (%)')
        ax3.set_title('Quantiles of Recent Returns')
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
{self.chain} - Recent Data Analysis
{'='*40}

Period: {result.period_days} days
Data Window: Last {result.recency_months} months
Date Range: {datetime.fromisoformat(result.start_date).date()} to {datetime.fromisoformat(result.end_date).date()}
Number of Periods: {result.num_periods}

Current TVL: ${result.current_tvl/1e6:.1f}M

Statistics:
- Mean Return: {(np.exp(result.log_returns_mean) - 1) * 100:.2f}%
- Volatility ({result.period_days}d): {result.volatility_period * 100:.1f}%
- Annualized Vol: {result.volatility_annual * 100:.1f}%

99% Confidence Bounds:
- Lower: ${result.lower_bound/1e6:.1f}M ({(result.lower_bound/result.current_tvl - 1) * 100:+.1f}%)
- Upper: ${result.upper_bound/1e6:.1f}M ({(result.upper_bound/result.current_tvl - 1) * 100:+.1f}%)
- Width: {result.upper_bound/result.lower_bound:.2f}x

Max Observed Extremes:
- Max Drawdown: {result.max_drawdown * 100:.1f}%
- Max Increase: {result.max_increase * 100:+.1f}%
- Max Lower: ${result.max_bounds_lower/1e6:.1f}M
- Max Upper: ${result.max_bounds_upper/1e6:.1f}M
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center', 
                family='monospace', transform=ax4.transAxes)
        
        plt.suptitle(f'Recent Data Bounds Analysis - {self.chain}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
    
    def print_summary(self, result: RecentBoundsResult) -> None:
        """Print summary of recent bounds calculation."""
        print("\n" + "="*60)
        print(f"RECENT DATA BOUNDS ANALYSIS FOR {result.chain.upper()}")
        print("="*60)
        
        print(f"\nAnalysis Parameters:")
        print(f"  Period: {result.period_days} days")
        print(f"  Using only last {result.recency_months} months of data")
        print(f"  Date range: {datetime.fromisoformat(result.start_date).date()} to {datetime.fromisoformat(result.end_date).date()}")
        print(f"  Number of periods: {result.num_periods}")
        
        print(f"\nCurrent TVL: ${result.current_tvl/1e6:.2f}M")
        
        print(f"\n{result.period_days}-Day Return Statistics:")
        print(f"  Mean: {(np.exp(result.log_returns_mean)-1)*100:.2f}%")
        print(f"  Volatility: {result.volatility_period*100:.1f}%")
        print(f"  Annualized: {result.volatility_annual*100:.1f}%")
        
        print(f"\nQuantiles:")
        print(f"  0.5%:  {(np.exp(result.quantile_005)-1)*100:+.1f}%")
        print(f"  2.5%:  {(np.exp(result.quantile_025)-1)*100:+.1f}%")
        print(f"  50%:   {(np.exp(result.quantile_50)-1)*100:+.1f}%")
        print(f"  97.5%: {(np.exp(result.quantile_975)-1)*100:+.1f}%")
        print(f"  99.5%: {(np.exp(result.quantile_995)-1)*100:+.1f}%")
        
        print(f"\n{result.confidence_level:.0%} CONFIDENCE BOUNDS:")
        print(f"  Lower: ${result.lower_bound/1e6:.2f}M ({(result.lower_bound/result.current_tvl-1)*100:+.1f}%)")
        print(f"  Upper: ${result.upper_bound/1e6:.2f}M ({(result.upper_bound/result.current_tvl-1)*100:+.1f}%)")
        print(f"  Width: {result.upper_bound/result.lower_bound:.2f}x")
        
        print(f"\nMAX OBSERVED EXTREMES (100% bounds):")
        print(f"  Max Drawdown: {result.max_drawdown*100:.1f}% → ${result.max_bounds_lower/1e6:.2f}M")
        print(f"  Max Increase: {result.max_increase*100:+.1f}% → ${result.max_bounds_upper/1e6:.2f}M")
        print(f"  Max Width: {result.max_bounds_upper/result.max_bounds_lower:.2f}x")
        
        print("="*60)


def main(chain: str = "Unichain", period_days: int = 21, recency_months: int = 3):
    """Main execution function."""
    
    # Initialize calculator
    calculator = RecentBoundsCalculator(
        chain=chain, 
        period_days=period_days, 
        recency_months=recency_months
    )
    
    # Calculate bounds
    result = calculator.calculate_bounds(confidence_level=0.99)
    
    # Print summary
    calculator.print_summary(result)
    
    # Save results
    results_dir = Path("portfolio_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save numerical results
    results_file = results_dir / f"{chain.lower()}_bounds_recent_{period_days}d_{recency_months}m.json"
    with open(results_file, 'w') as f:
        result_dict = asdict(result)
        result_dict['analysis_timestamp'] = datetime.now().isoformat()
        json.dump(result_dict, f, indent=2)
    print(f"\nSaved results to {results_file}")
    
    # Visualize
    viz_file = results_dir / f"{chain.lower()}_bounds_recent_{period_days}d_{recency_months}m_viz.png"
    calculator.visualize_analysis(result, save_path=str(viz_file))
    
    return result


if __name__ == "__main__":
    import sys
    chain = sys.argv[1] if len(sys.argv) > 1 else "Unichain"
    period_days = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    recency_months = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    result = main(chain, period_days, recency_months)