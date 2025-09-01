#!/usr/bin/env python3
"""
Comprehensive comparison of bounds across all chains.
Uses 12-month data for Base/Arbitrum, 3-month data for Unichain.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


def load_chain_data(chain: str, period: int = 21, recency_months: int = None) -> Dict:
    """Load bounds data for a specific chain."""
    results_dir = Path("../scalar_bounds_analysis/results")
    
    if recency_months:
        # Load recent bounds data
        file_path = results_dir / f"{chain.lower()}_bounds_recent_{period}d_{recency_months}m.json"
    else:
        # Load standard scalar bounds
        file_path = results_dir / f"{chain.lower()}_scalar_bounds_{period}d.json"
    
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    
    # Try without period suffix for backward compatibility
    file_path = results_dir / f"{chain.lower()}_scalar_bounds.json"
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    
    return None


def create_comprehensive_comparison():
    """Create comprehensive comparison chart for all chains."""
    
    # Load data for each chain with appropriate settings
    chains_data = {
        'Base': load_chain_data('Base', 21, 12),      # 12-month data
        'Arbitrum': load_chain_data('Arbitrum', 21, 12),  # 12-month data
        'Unichain': load_chain_data('Unichain', 21, 3)    # 3-month data only
    }
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid spec for custom layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main comparison chart (top row, spanning 2 columns)
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Statistics table (top right)
    ax_table = fig.add_subplot(gs[0, 2])
    
    # Individual volatility charts (middle row)
    ax_vol = fig.add_subplot(gs[1, :])
    
    # Quantile charts (bottom row)
    ax_quantiles = fig.add_subplot(gs[2, :])
    
    # Colors for each chain
    colors = {'Base': '#1f77b4', 'Arbitrum': '#ff7f0e', 'Unichain': '#2ca02c'}
    
    # 1. Main Bounds Comparison
    chains = list(chains_data.keys())
    x_pos = np.arange(len(chains))
    
    for i, (chain, data) in enumerate(chains_data.items()):
        if data:
            current = data['current_tvl'] / 1e9
            lower = data['lower_bound'] / 1e9
            upper = data['upper_bound'] / 1e9
            
            # Plot error bar style bounds
            ax_main.plot([i, i], [lower, upper], 'o-', color=colors[chain], 
                        linewidth=3, markersize=10, label=chain)
            
            # Add current value marker
            ax_main.plot(i, current, 's', color=colors[chain], markersize=8, 
                        markeredgecolor='black', markeredgewidth=1)
            
            # Add value labels
            ax_main.text(i, upper + 0.1, f'${upper:.2f}B', ha='center', fontsize=9, 
                        fontweight='bold')
            ax_main.text(i, lower - 0.1, f'${lower:.2f}B', ha='center', fontsize=9, 
                        fontweight='bold')
            ax_main.text(i + 0.02, current, f'${current:.2f}B', ha='left', fontsize=8, 
                        style='italic', bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor="white", alpha=0.8))
            
            # Add width label
            width = upper / lower
            mid = (upper + lower) / 2
            ax_main.text(i + 0.15, mid, f'{width:.2f}x', ha='left', fontsize=9, 
                        color='gray', style='italic')
    
    ax_main.set_xticks(x_pos)
    ax_main.set_xticklabels(chains)
    ax_main.set_ylabel('TVL (Billions USD)', fontsize=11)
    ax_main.set_title('21-Day TVL Bounds Comparison (99% CI)', fontsize=13, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper left')
    
    # 2. Statistics Table
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table_data = [['Chain', 'Period', 'Vol', 'Width', 'Max Loss', 'Max Gain']]
    
    for chain, data in chains_data.items():
        if data:
            period_days = data.get('period_days', 21)
            recency = data.get('recency_months', 12)
            volatility = data.get('volatility_period', 0) * 100
            lower_pct = (data['lower_bound'] / data['current_tvl'] - 1) * 100
            upper_pct = (data['upper_bound'] / data['current_tvl'] - 1) * 100
            width = data['upper_bound'] / data['lower_bound']
            
            # Get max observed if available
            max_down = data.get('max_drawdown', 0) * 100
            max_up = data.get('max_increase', 0) * 100
            
            if chain == 'Unichain':
                period_str = f'{period_days}d (3m)'
            else:
                period_str = f'{period_days}d (12m)'
            
            table_data.append([
                chain,
                period_str,
                f'{volatility:.1f}%',
                f'{width:.2f}x',
                f'{max_down:.1f}%' if max_down != 0 else f'{lower_pct:.1f}%',
                f'{max_up:+.1f}%' if max_up != 0 else f'{upper_pct:+.1f}%'
            ])
    
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(weight='bold')
    
    # 3. Volatility Comparison
    vol_data = []
    vol_labels = []
    vol_colors = []
    
    for chain, data in chains_data.items():
        if data:
            vol_data.append(data.get('volatility_period', 0) * 100)
            vol_labels.append(chain)
            vol_colors.append(colors[chain])
    
    bars = ax_vol.bar(range(len(vol_labels)), vol_data, color=vol_colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars, vol_data):
        height = bar.get_height()
        ax_vol.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                   fontweight='bold')
    
    ax_vol.set_xticks(range(len(vol_labels)))
    ax_vol.set_xticklabels(vol_labels)
    ax_vol.set_ylabel('21-Day Volatility (%)', fontsize=11)
    ax_vol.set_title('Volatility Comparison', fontsize=12, fontweight='bold')
    ax_vol.grid(True, alpha=0.3, axis='y')
    
    # Add data period annotations
    ax_vol.text(0, vol_data[0] * 0.5, '12 months', ha='center', fontsize=8, color='white', fontweight='bold')
    ax_vol.text(1, vol_data[1] * 0.5, '12 months', ha='center', fontsize=8, color='white', fontweight='bold')
    ax_vol.text(2, vol_data[2] * 0.5, '3 months', ha='center', fontsize=8, color='white', fontweight='bold')
    
    # 4. Quantile Comparison
    quantiles = ['0.5%', '2.5%', '50%', '97.5%', '99.5%']
    x = np.arange(len(quantiles))
    width = 0.25
    
    for i, (chain, data) in enumerate(chains_data.items()):
        if data and 'quantile_005' in data:
            quantile_values = [
                (np.exp(data.get('quantile_005', 0)) - 1) * 100,
                (np.exp(data.get('quantile_025', 0)) - 1) * 100,
                (np.exp(data.get('quantile_50', 0)) - 1) * 100,
                (np.exp(data.get('quantile_975', 0)) - 1) * 100,
                (np.exp(data.get('quantile_995', 0)) - 1) * 100,
            ]
            
            bars = ax_quantiles.bar(x + i * width, quantile_values, width, 
                                   label=chain, color=colors[chain], alpha=0.7)
            
            # Add value labels for extreme quantiles
            for j, (pos, val) in enumerate(zip(x + i * width, quantile_values)):
                if j in [0, 4]:  # Only label extremes
                    ax_quantiles.text(pos, val, f'{val:.1f}%', ha='center', 
                                    va='bottom' if val > 0 else 'top', 
                                    fontsize=7, rotation=45)
    
    ax_quantiles.set_xlabel('Quantiles', fontsize=11)
    ax_quantiles.set_ylabel('21-Day Return (%)', fontsize=11)
    ax_quantiles.set_title('Return Distribution Quantiles', fontsize=12, fontweight='bold')
    ax_quantiles.set_xticks(x + width)
    ax_quantiles.set_xticklabels(quantiles)
    ax_quantiles.legend()
    ax_quantiles.grid(True, alpha=0.3, axis='y')
    ax_quantiles.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Overall title
    fig.suptitle('Comprehensive TVL Bounds Analysis - All Chains', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Add footnote
    fig.text(0.5, 0.01, 
            'Note: Base and Arbitrum use 12-month historical data. Unichain uses 3-month data due to limited history.',
            ha='center', fontsize=9, style='italic')
    
    # Save figure
    output_path = Path("portfolio_results") / "all_chains_comprehensive_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comprehensive comparison to {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPREHENSIVE BOUNDS SUMMARY (21-DAY)")
    print("="*80)
    print(f"{'Chain':<12} {'Data Period':<12} {'Current':<10} {'Lower Bound':<15} {'Upper Bound':<15} {'Width':<8} {'Volatility':<10}")
    print("-"*80)
    
    for chain, data in chains_data.items():
        if data:
            current = data['current_tvl'] / 1e9
            lower = data['lower_bound'] / 1e9
            upper = data['upper_bound'] / 1e9
            width = upper / lower
            vol = data.get('volatility_period', 0) * 100
            
            if chain == 'Unichain':
                period = '3 months'
            else:
                period = '12 months'
            
            lower_pct = (data['lower_bound'] / data['current_tvl'] - 1) * 100
            upper_pct = (data['upper_bound'] / data['current_tvl'] - 1) * 100
            
            print(f"{chain:<12} {period:<12} ${current:.2f}B     "
                  f"${lower:.2f}B ({lower_pct:+.1f}%)  "
                  f"${upper:.2f}B ({upper_pct:+.1f}%)  "
                  f"{width:.2f}x   {vol:.1f}%")
    
    print("="*80)
    
    return chains_data


if __name__ == "__main__":
    # First ensure we have the necessary data
    import subprocess
    import sys
    
    print("Generating required data files...")
    
    # Run Base and Arbitrum with 12-month data
    subprocess.run([sys.executable, "src/scalar_bounds_recent.py", "Base", "21", "12"])
    subprocess.run([sys.executable, "src/scalar_bounds_recent.py", "Arbitrum", "21", "12"])
    
    # Unichain with 3-month data is already available
    # subprocess.run([sys.executable, "src/scalar_bounds_recent.py", "Unichain", "21", "3"])
    
    # Create the comprehensive comparison
    chains_data = create_comprehensive_comparison()