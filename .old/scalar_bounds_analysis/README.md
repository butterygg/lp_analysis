# Scalar Bounds Analysis

This module calculates prediction market bounds for chain TVL using advanced statistical methods.

## Files
- `scalar_bounds_calculator.py` - Main bounds calculation with empirical methods
- `scalar_bounds_recent.py` - Recent period focused analysis  
- `scalar_bounds_unichain_3m.py` - Specialized for Unichain with 3-month data
- `results/` - Output directory for bounds calculations

## Usage

From the repository root:
```bash
# Calculate bounds for Base chain, 30-day periods, 12-month lookback
python run_scalar_bounds.py Base 30 12

# Calculate bounds for Unichain with 3-month data
python run_scalar_bounds.py Unichain 21 3
```

Or directly:
```bash
cd scalar_bounds_analysis
python scalar_bounds_calculator.py [chain] [period_days] [lookback_months]
```

## Features
- Log-returns instead of arithmetic returns
- Empirical quantiles instead of normal distribution assumptions
- Bootstrap confidence intervals
- Regime-aware volatility analysis
- Cross-sectional priors for young chains
- Age-based caps for new chains

## Outputs
- JSON files with detailed bounds calculations
- PNG visualization comparing different bound methods
- Console summary with statistics and methodology details