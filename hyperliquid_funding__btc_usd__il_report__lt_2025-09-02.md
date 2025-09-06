# LP IL-Only Report — `hyperliquid_funding` / `btc_usd`

_Windows strictly earlier than **2025-09-02**. All figures exclude trading fees._

## Data Source
**Profile Question:** Use Hyperliquid info fundingHistory at https://api.hyperliquid.xyz/info (type=fundingHistory, coin=BTC). For each UTC day compute daily_APY_percent = (sum of that day's funding rates as fractions) * 365 * 100. Return the arithmetic mean of these daily_APY_percent values for records with {MARKET_START_DATE_UTC} < t ≤ {MARKET_END_DATE_UTC}, in basis points.

## Market Structure
Each market contains **UP** and **DOWN** tokens representing directional bets on changes in the underlying metric:
- **UP tokens** increase in value when the metric rises above its starting level
- **DOWN tokens** increase in value when the metric falls below its starting level
- Token prices are obtained by a piece-wise linear mapping of the metric into a bounded range; the UP price is the scaled value and the DOWN price is `1 − UP`
- UP and DOWN prices always sum to **$1.00**, forming a complementary pair

## Liquidity Pool Mechanics
- **Initialization**: UP and DOWN are minted in **equal proportion** before depositing to the pool
- **AMM model**: Constant product (Uniswap v2 style) with equal notional value deposited initially
- **Ratio mismatch & leftovers**: Because the AMM requires a specific price ratio, one side is typically over-minted. The excess remains **un-deposited** ("leftover") and is carried passively to the end of the period
- **Pricing updates**: Prices evolve as the underlying metric changes; UP and DOWN continue to sum to $1.00

# LP Return Distribution
We simulate starting at each historical window strictly earlier than the cutoff date. 
We **exclude** very early windows until a minimum history (processing.min_historical_data_months) has elapsed to avoid unstable bounds.

## Important
- **Mean** -8.66% and **median** -4.30% IL-only returns are shown below.
- These IL losses must be compared to incentive APY to calculate your net returns.

## Portfolio Performance
### IL Distribution Histogram
![Portfolio Return Distributions](hyperliquid_funding__btc_usd__il_hist__lt_2025-09-02.png)

### IL Over Time
![IL Returns Over Time](hyperliquid_funding__btc_usd__il_timeseries__lt_2025-09-02.png)

This time series shows how IL-only portfolio returns have varied across different historical windows.

### Distribution Summary (IL-only, %)

- Count: **228**
- Mean: **-8.66%**, Std: **10.37%**
- Median: **-4.30%**  |  P25: **-12.76%**  |  P10: **-26.95%**  |  P75: **-0.73%**

## Calculating Your Net APY

To determine your actual returns, combine Merkl incentive APY with these IL losses:

**Period Factor**: 0.063 (since this is a 23-day market)

### Formula:
```
Period_Factor = Market_Duration_Days / 365
Incentive_Multiplier = 1 + (Merkl_APY × Period_Factor)
Net_Return = Incentive_Multiplier × (1 + IL_Return) - 1
```

Where:
- **Period_Factor**: Scales annual incentive APY to the market duration
- **Merkl_APY**: The annual percentage yield from Merkl incentives (as a decimal)
- **IL_Return**: Your expected impermanent loss return (as a decimal, typically negative)

### Example Calculation (Hypothetical Numbers Only):
**Example calculation only**: Let's say Merkl shows **200% APY** (this is just an example - actual APY varies by market) and you experience the **median IL loss (-4.30%)**:

1. **Scale Merkl APY to period**: 200% × 0.063 = 12.6%
2. **Convert to multiplier**: 1 + 12.6% = 1.126
3. **Apply median IL loss**: 1.126 × (1 + -4.3%) = 1.126 × 0.957 = 1.078
4. **Net return for 23 days**: 7.8%
5. **Annualized (APY)**: (1.078)^15.9 - 1 = **227.4% APY**

**Steps to use this with your actual numbers:**
1. Find your market's Merkl campaign and note the **actual APY** (not the 200% example)
   - **Note**: Merkl APY can vary over the duration of the market depending on the amount of liquidity provided
2. Multiply that APY by **0.063**
3. Add 1 to get the incentive multiplier
4. Multiply by (1 + your_expected_IL_return)
5. Subtract 1 to get your net return over 23 days
6. To annualize: raise (1 + return) to the power of 15.9, then subtract 1

# Technical Implementation

## Outputs
- **CSV (per-window IL)**: [hyperliquid_funding__btc_usd__il_by_window__lt_2025-09-02.csv](hyperliquid_funding__btc_usd__il_by_window__lt_2025-09-02.csv)

# Disclaimer
This analysis is for informational purposes only and does not constitute financial advice. Results are based on historical data and may not reflect future performance. Simulation code and models may contain errors or inaccuracies.