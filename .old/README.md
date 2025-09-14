# Liquidity Provider Analysis Repository

## Overview  
This repository contains modular analysis tools for liquidity providers (LPs) in DeFi markets, including:
- Uniswap V3 impermanent loss analysis
- Multi-chain portfolio simulations
- Scalar bounds calculations for prediction markets
- Cross-chain comparison tools
- Professional visualizations

The repository is organized into self-contained modules that can be run independently.

## Repository Structure

```
lp_analysis/
├── uniswap_analysis/          # Uniswap V3 IL/Volume analysis
├── portfolio_simulation/      # Multi-chain LP portfolio simulation
├── scalar_bounds_analysis/    # Prediction market bounds calculation
├── comparison_tools/          # Cross-chain and method comparisons
├── visualization/            # Advanced visualization tools
├── cache/                    # Shared API data cache
├── config.json              # Configuration file
└── run_*.py                 # Convenience wrapper scripts
```

## Quick Start

Run any module from the repository root:

```bash
# Analyze Uniswap V3 pools
python run_uniswap_analysis.py --help

# Run portfolio simulation
python run_portfolio_simulation.py

# Calculate scalar bounds for prediction markets
python run_scalar_bounds.py Base 30 12

# Compare across chains
python run_compare_chains.py

# Generate visualizations
python run_visualization.py
```

Each module can also be run directly from its subdirectory. See individual module READMEs for detailed usage.

## Module Overview

### Uniswap Analysis
Analyzes impermanent loss vs trading volume for Uniswap V3 pools using subgraph data.

### Portfolio Simulation  
Simulates LP returns across multiple chains using rolling time windows with fee, IL, and external return components.

### Scalar Bounds Analysis
Calculates prediction market bounds for chain TVL using empirical methods, log-returns, and bootstrap confidence intervals.

### Comparison Tools
Provides cross-chain comparisons and methodology evolution tracking.

### Visualization
Creates professional visualizations of LP return distributions and analysis results.

## Original Research

The detailed methodology and findings from the original research are preserved in [REPORT.md](.old/REPORT.md).