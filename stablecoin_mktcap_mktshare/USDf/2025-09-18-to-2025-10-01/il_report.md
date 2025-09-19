# LP IL-Only Report — `stablecoin_mktcap_mktshare` / `USDf`

_Windows strictly earlier than **2025-09-18**. All figures exclude trading fees._

## Data Source
**Metric which market is forecasting:** USDf market-cap share as of Oct 1, 2025

<details><summary>Oracle Question used to resolve metric value</summary>

Use the DefiLlama stablecoin endpoint at https://stablecoins.llama.fi/stablecoin/246. Extract the circulating supply (pegged USD) series and return the value with the greatest timestamp at or before 2025-10-01 00:00:00 UTC (UTC). For the denominator, query https://stablecoins.llama.fi/stablecoin/<id> for each of the following USD-pegged stablecoins and sum their circulating peggedUSD values at that same timestamp: Tether (USDT, id 1); USD Coin (USDC, id 2); Ethena USDe (USDe, id 146); Sky Dollar (USDS, id 209); World Liberty Financial USD (USD1, id 262); BlackRock USD (BUIDL, id 173); Ethena USDtb (USDTB, id 221); Falcon USD (USDf, id 246); PayPal USD (PYUSD, id 120); First Digital USD (FDUSD, id 119); Ripple USD (RLUSD, id 250); USDX Money USDX (USDX, id 214). Compute the market-cap share for that stablecoin (the one identified by stablecoin_id above) as (stablecoin circulating / aggregate circulating) * 100 and report that percentage * 100, rounded up to the nearest integer. 

</details>

## Market Structure
Each market contains **UP** and **DOWN** tokens representing directional bets on changes in the underlying metric:
- **UP tokens** increase in value when the metric rises above its starting level
- **DOWN tokens** increase in value when the metric falls below its starting level
- Token prices are obtained by a piece-wise linear mapping of the metric into a bounded range; the UP price is the scaled value and the DOWN price is `1 − UP`
- UP and DOWN prices always sum to **$1.00**, forming a complementary pair

## Price Mapping
- Market bounds: **min = 0.2**, **max = 1.5**. UP's USD price p is a linear mapping of the metric m into [0,1].
- Mapping: we scale the metric between min and max to get a number p between 0 and 1 (values below min map to 0; above max map to 1)
- DOWN's USD price is 1 − p
- AMM pool price (UP:DOWN) = p / (1 − p)
- Intuition: when the metric rises (p ↑), UP becomes more valuable relative to DOWN, so the pool price p/(1−p) increases; the LP's inventory rebalances toward DOWN and vice versa
- Impermanent loss depends on how far the pool price moves away from the starting price at your deposit; larger moves ⇒ larger IL (fees excluded here)

### Worked Example (for intuition)
- Take m at 60% of range: m = min + 0.60 × (max − min) = 0.98
- UP price: p = (m − min) / (max − min) = (0.98 − 0.2) / (1.5 − 0.2) = **0.600**
- AMM pool price (UP:DOWN): p/(1 − p) = 0.600 / 0.400 = **1.500**


## Liquidity Pool Mechanics
- **Initialization**: UP and DOWN are minted in **equal proportion** before depositing to the pool
- **AMM model**: Constant product (Uniswap v2 style) with equal notional value deposited initially
- **Ratio mismatch & leftovers**: Because the AMM requires a specific price ratio, one side is typically over-minted. The excess remains **un-deposited** ("leftover") and is carried passively to the end of the period
- **Pricing updates**: Prices evolve as the underlying metric changes; UP and DOWN continue to sum to $1.00

# LP Return Distribution
We simulate starting at each historical window strictly earlier than the cutoff date. 
We **exclude** very early windows until a minimum history (processing.min_historical_data_months) has elapsed to avoid unstable bounds.

## Important
- **Mean** -11.47% and **median** -1.48% IL-only returns are shown below.
- These IL losses must be compared to incentive APY to calculate your net returns.

## Portfolio Performance
### IL Distribution Histogram
![Portfolio Return Distributions](il_hist.png)

### IL Over Time
![IL Returns Over Time](il_timeseries.png)

This time series shows how IL-only portfolio returns have varied across different historical windows.

### Distribution Summary (IL-only, %)

- Count: **105**
- Mean: **-11.47%**, Std: **24.93%**
- Median: **-1.48%**  |  P25: **-5.18%**  |  P10: **-55.06%**  |  P75: **-0.28%**

## Calculating Your Net APY

To determine your actual returns, combine Merkl incentive APY with these IL losses:

**Period Factor**: 0.036 (since this is a 13-day market)

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
**Example calculation only**: Let's say Merkl shows **200% APY** (this is just an example - actual APY varies by market) and you experience the **median IL loss (-1.48%)**:

1. **Scale Merkl APY to period**: 200% × 0.036 = 7.1%
2. **Convert to multiplier**: 1 + 7.1% = 1.071
3. **Apply median IL loss**: 1.071 × (1 + -1.5%) = 1.071 × 0.985 = 1.055
4. **Net return for 13 days**: 5.5%
5. **Annualized (APY)**: (1.055)^28.1 - 1 = **354.4% APY**

**Steps to use this with your actual numbers:**
1. Find your market's Merkl campaign and note the **actual APY** (not the 200% example)
   - **Note**: Merkl APY can vary over the duration of the market depending on the amount of liquidity provided
2. Multiply that APY by **0.036**
3. Add 1 to get the incentive multiplier
4. Multiply by (1 + your_expected_IL_return)
5. Subtract 1 to get your net return over 13 days
6. To annualize: raise (1 + return) to the power of 28.1, then subtract 1

# Technical Implementation

## Outputs
- **CSV (per-window IL)**: [il_by_window.csv](il_by_window.csv)

# Disclaimer
This analysis is for informational purposes only and does not constitute financial advice. Results are based on historical data and may not reflect future performance. Simulation code and models may contain errors or inaccuracies.