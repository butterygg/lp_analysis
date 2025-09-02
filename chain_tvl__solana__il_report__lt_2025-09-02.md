# LP IL-Only Report — `chain_tvl` / `solana`

_Windows strictly earlier than **2025-09-02**. All figures exclude trading fees._

## Data Source
**Profile Description:** Solana TVL from https://api.llama.fi/v2/historicalChainTvl/ using chain 'Solana', taking the last TVL value within each 30-day window

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
We simulate starting at each historical window strictly earlier than the cutoff date. The **start price** for each window is built from the mean of the last **5** metric observations available up to **exactly one period** (**30 days**) **before** that window's last date. The **end price** is mapped from the window's own metric.
We **exclude** very early windows until a minimum history (processing.min_il_calc_history_months) has elapsed to avoid unstable bounds.

## Important
- **Mean** -14.25% and **median** -4.73% IL-only returns are shown below.
- These IL losses must be compared to incentive APY to calculate your net returns.

## Portfolio Performance
### IL Distribution Histogram
![Portfolio Return Distributions](chain_tvl__solana__il_hist__lt_2025-09-02.png)

### IL Over Time
![IL Returns Over Time](chain_tvl__solana__il_timeseries__lt_2025-09-02.png)

This time series shows how IL-only portfolio returns have varied across different historical windows.

### Distribution Summary (IL-only, %)

- Count: **399**
- Mean: **-14.25%**, Std: **22.27%**
- Median: **-4.73%**  |  P25: **-17.07%**  |  P10: **-42.53%**  |  P75: **-0.73%**

## Calculating Your Net APY

To determine your actual returns, combine Merkl incentive APY with these IL losses:

**Period Factor**: 0.082 (since this is a 30-day market)

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
**Example calculation only**: Let's say Merkl shows **200% APY** (this is just an example - actual APY varies by market) and you experience the **median IL loss (-4.73%)**:

1. **Scale Merkl APY to period**: 200% × 0.082 = 16.4%
2. **Convert to multiplier**: 1 + 16.4% = 1.164
3. **Apply median IL loss**: 1.164 × (1 + -4.7%) = 1.164 × 0.953 = 1.109
4. **Net return for 30 days**: 10.9%
5. **Annualized (APY)**: (1.109)^12.2 - 1 = **253.5% APY**

**Steps to use this with your actual numbers:**
1. Find your market's Merkl campaign and note the **actual APY** (not the 200% example)
   - **Note**: Merkl APY can vary over the duration of the market depending on the amount of liquidity provided
2. Multiply that APY by **0.082**
3. Add 1 to get the incentive multiplier
4. Multiply by (1 + your_expected_IL_return)
5. Subtract 1 to get your net return over 30 days
6. To annualize: raise (1 + return) to the power of 12.2, then subtract 1

# Technical Implementation

## Outputs
- **CSV (per-window IL)**: [chain_tvl__solana__il_by_window__lt_2025-09-02.csv](chain_tvl__solana__il_by_window__lt_2025-09-02.csv)

# Disclaimer
This analysis is for informational purposes only and does not constitute financial advice. Results are based on historical data and may not reflect future performance. Simulation code and models may contain errors or inaccuracies.