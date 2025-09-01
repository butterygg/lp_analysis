# DeFi LP Portfolio Simulation

This module simulates liquidity provider returns across multiple chains using rolling windows.

## Files
- `defi_lp_portfolio_simulation.py` - Main simulation orchestrator
- `lp_simulation_utils.py` - Utility classes and functions
- `results/` - Output directory for simulation results
- `lp_returns/` - Return distribution visualizations by chain

## Usage

From the repository root:
```bash
python run_portfolio_simulation.py
```

Or directly:
```bash
cd portfolio_simulation
python defi_lp_portfolio_simulation.py
```

## Outputs
- `results/simulation_results.json` - Comprehensive simulation results
- Individual chain return distributions (PNG files)
- Console summary statistics and comparisons

## Features
- Multi-chain portfolio simulation (Arbitrum, Base, Unichain, Hyperliquid L1)
- Rolling 30-day simulation windows
- Fee, IL, and external return component tracking
- Bootstrap and statistical analysis