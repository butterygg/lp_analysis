# Comparison Tools

This module provides tools for comparing results across different chains and methodologies.

## Files
- `compare_all_chains.py` - Cross-chain bounds comparison
- `compare_bounds_methods.py` - Old vs new methodology comparison
- `compare_periods.py` - Different time period analysis

## Usage

From the repository root:
```bash
python run_compare_chains.py
```

Or directly:
```bash
cd comparison_tools
python compare_all_chains.py
```

## Features
- Side-by-side chain comparisons
- Methodology evolution tracking
- Statistical significance testing
- Comprehensive visualization charts

## Dependencies
- Requires results from `scalar_bounds_analysis` module
- May require results from `portfolio_simulation` for some comparisons