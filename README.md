# Scalar Bounds Calculator for Prediction Markets

Enhanced TVL bounds calculation for DeFi chains, optimized for Uniswap v2 scalar prediction markets based on recommendations from `scalar-bounds-design.md`.

## Setup

1. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install numpy scipy matplotlib requests
```

## Main Scripts

### 1. Scalar Bounds Calculator (PRIMARY)
Calculate bounds using empirical quantiles and log-returns for any period:

```bash
# Default 30-day bounds
python src/scalar_bounds_calculator.py Base
python src/scalar_bounds_calculator.py Arbitrum
python src/scalar_bounds_calculator.py Unichain

# Custom period (e.g., 20-day, 21-day)
python src/scalar_bounds_calculator.py Base 20
python src/scalar_bounds_calculator.py Arbitrum 21
```

**Features:**
- **Log-returns** instead of arithmetic returns for better multiplicative growth handling
- **Empirical quantiles** (actual 0.5%/99.5%) instead of assuming normal distribution
- **Bootstrap confidence intervals** with block sampling to preserve autocorrelation
- **Regime-aware bounds** that adapt to calm vs stress market conditions
- **Automatic young series handling** with cross-chain priors and smart caps
- **Configurable period** (default 30 days, but supports any period)

### 2. Recent Data Bounds Calculator
Calculate bounds using only recent historical data:

```bash
# Using last 3 months for 21-day bounds
python src/scalar_bounds_recent.py Unichain 21 3

# Using last 12 months for 21-day bounds
python src/scalar_bounds_recent.py Base 21 12
python src/scalar_bounds_recent.py Arbitrum 21 12
```

**Features:**
- **Recency filter** to focus on recent market conditions
- **Max observed bounds** showing actual historical extremes
- **Detailed quantile analysis** (0.5%, 2.5%, 50%, 97.5%, 99.5%)
- **Comparison visualization** between recent and full history

### 3. Period Comparison
Compare bounds across different time periods:

```bash
# Compare 20-day vs 30-day bounds
python src/compare_periods.py
```

### 4. Method Comparison
Compare old (normal distribution) vs new (empirical) methods:

```bash
python src/compare_bounds_methods.py
```

## Output Files

Results are saved in `portfolio_results/`:

**Scalar bounds outputs:**
- `{chain}_scalar_bounds_{period}d.json` - Numerical bounds and statistics
- `{chain}_scalar_bounds_{period}d_viz.png` - Multi-panel visualization

**Recent bounds outputs:**
- `{chain}_bounds_recent_{period}d_{months}m.json` - Recent window analysis
- `{chain}_bounds_recent_{period}d_{months}m_viz.png` - Visualization with max bounds

**Comparison outputs:**
- `period_comparison.png` - Comparison across different periods
- `bounds_methods_comparison.png` - Old vs new method comparison

## Key Improvements

The scalar bounds calculator addresses critical issues from the design document:

1. **Statistical Robustness**
   - Log-returns: `log(M_{t+30}/M_t)` vs old `(M_{t+30}-M_t)/M_t`
   - Empirical quantiles from actual data vs z-score assumptions
   - Bootstrap and regime detection for market conditions

2. **Young Series Handling**
   - Cross-sectional priors from mature chains (Base, Arbitrum)
   - Age-based caps: 5x (<90 days), 3x (<180 days), 2x (>180 days)
   - Growth-rate caps: 2.5x max if weekly growth >50%
   - Automatic prior weighting based on data availability

3. **Prediction Market Optimization**
   - Bounds designed to avoid Uniswap v2 "dead zones" near extremes
   - Tighter, more tradeable ranges from empirical methods
   - Multiple methods provide validation and confidence

## Example Results

### 30-Day Bounds (Full 12-month history)
| Chain        | Current TVL | 99% CI Bounds    | Width | Volatility      |
| ------------ | ----------- | ---------------- | ----- | --------------- |
| **Base**     | $4.62B      | [$3.71B, $6.92B] | 1.9x  | 16.1%           |
| **Arbitrum** | $3.20B      | [$2.57B, $4.47B] | 1.7x  | 13.7%           |
| **Unichain** | $0.69B      | [$0.53B, $2.07B] | 3.9x  | 335.1% (capped) |

### 21-Day Bounds (Recent 3 months only)
| Chain        | Current TVL | 99% CI Bounds    | Width | Max Observed    |
| ------------ | ----------- | ---------------- | ----- | --------------- |
| **Base**     | $4.62B      | [$4.44B, $5.97B] | 1.35x | -4.2% / +30.6%  |
| **Arbitrum** | $3.20B      | [$2.91B, $3.80B] | 1.30x | -9.4% / +19.6%  |
| **Unichain** | $0.69B      | [$0.54B, $1.11B] | 2.06x | -21.8% / +61.0% |

## Cached Data

TVL data is cached in `cache/` directory:
- `chain_tvl_{Chain}.json` - Historical TVL data from DeFiLlama
- Delete cache files to force fresh data fetch

## Deprecated Scripts

Old scripts using arithmetic returns and normal distribution have been moved to `src/_old/`:
- `tvl_bounds_calculator.py` - Original normal distribution method
- `tvl_bounds_young_chains.py` - Separate young chains handler
- `compare_chain_bounds.py` - Old comparison script

These are retained for reference but should not be used for production bounds setting.