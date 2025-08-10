# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scalar bounds calculator for DeFi prediction markets, optimized for Uniswap v2. It calculates TVL bounds for chains using empirical quantiles and log-returns, addressing issues identified in `scalar-bounds-design.md`. The project fetches historical data from DeFiLlama API and provides confidence intervals for prediction market parameter setting.

## Architecture

### Primary Scripts (Active)
- **`src/scalar_bounds_calculator.py`** - Main bounds calculator using empirical quantiles and log-returns
- **`src/scalar_bounds_recent.py`** - Bounds calculator with recency filtering and max observed extremes
- **`src/compare_periods.py`** - Comparison tool for different time periods (20-day vs 30-day)
- **`src/compare_bounds_methods.py`** - Comparison of old (normal) vs new (empirical) methods

### Supporting Files
- **`src/lp_simulation_utils.py`** - Shared utilities for data fetching and caching
- **`scalar-bounds-design.md`** - Design document with requirements and recommendations
- **Data caching**: `cache/` directory - JSON files with historical chain TVL data from DeFiLlama
- **Results output**: `portfolio_results/` directory - Bounds calculations and visualizations

### Deprecated (in `src/_old/`)
- `tvl_bounds_calculator.py` - Old arithmetic returns + normal distribution method
- `tvl_bounds_young_chains.py` - Old separate young chains handler
- LP simulation scripts - Original portfolio simulation code

## Key Components

### Bounds Calculation Methods
- **Empirical Quantiles**: Direct 0.5%/99.5% percentiles from historical log-returns
- **Bootstrap**: Block bootstrap with 1000 iterations for autocorrelation-aware bounds
- **Regime-Aware**: Separate calm/stress periods based on volatility threshold
- **Max Observed**: Actual historical extremes for worst-case bounds

### Young Series Handling
- Cross-sectional priors from mature chains (Base, Arbitrum)
- Age-based caps: 5x (<90 days), 3x (<180 days), 2x (>180 days)
- Growth-rate caps: 2.5x if weekly growth >50%
- Bayesian-style blending with prior weight based on data availability

### Data Flow
1. Chain TVL data fetching from DeFiLlama API
2. Local caching in `cache/` directory
3. Log-returns calculation for specified period
4. Empirical quantile computation
5. Young series adjustments if applicable
6. Bounds output with visualization

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -e .

# Or using specific Python version
python3.13 -m pip install -e .
```

### Running Bounds Calculations
```bash
# Standard 30-day bounds
python src/scalar_bounds_calculator.py Base
python src/scalar_bounds_calculator.py Arbitrum
python src/scalar_bounds_calculator.py Unichain

# Custom period (20-day, 21-day, etc.)
python src/scalar_bounds_calculator.py Base 21
python src/scalar_bounds_calculator.py Arbitrum 20

# Recent data only (21-day bounds, last 3 months)
python src/scalar_bounds_recent.py Unichain 21 3

# Using last 12 months
python src/scalar_bounds_recent.py Base 21 12

# Comparisons
python src/compare_periods.py          # 20-day vs 30-day
python src/compare_bounds_methods.py   # Old vs new methods
```

### Key Parameters
- **Period**: Number of days for bounds calculation (default: 30)
- **Lookback**: Historical data window in months (default: 12)
- **Confidence Level**: Typically 99% (1% probability outside bounds)
- **Recency Window**: For recent-only analysis (3 or 12 months typical)

## Dependencies

- **requests**: DeFiLlama API communication
- **numpy**: Numerical computations for AMM math
- **matplotlib**: Performance visualization
- **python-dotenv**: Environment variable management
- **loguru**: Structured logging
- **PySnooper**: Debug tracing (optional)

## Type Checking

Type checking is disabled in `pyrightconfig.json` (`"typeCheckingMode": "off"`). The codebase uses type hints for documentation but doesn't enforce strict typing.

## Output Files

### Bounds Calculations
- `portfolio_results/{chain}_scalar_bounds_{period}d.json`: Bounds with all statistics
- `portfolio_results/{chain}_scalar_bounds_{period}d_viz.png`: Multi-panel visualization
- `portfolio_results/{chain}_bounds_recent_{period}d_{months}m.json`: Recent window analysis
- `portfolio_results/{chain}_bounds_recent_{period}d_{months}m_viz.png`: Recent data visualization

### Comparisons
- `portfolio_results/period_comparison.png`: Period comparison visualization
- `portfolio_results/bounds_methods_comparison.png`: Method comparison chart

### Cache
- `cache/chain_tvl_{Chain}.json`: Cached TVL data from DeFiLlama API