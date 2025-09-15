# LP IL-Only Report — `cmc_price` / `hype`

_Windows strictly earlier than **2025-09-02**. All figures exclude trading fees._

## Data Source
**Metric which market is forecasting:** HYPE price (USD) - daily median on Sep 26, 2025

<details><summary>Oracle Question used to resolve metric value</summary>

Use the CoinMarketCap detail/chart endpoint at https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail/chart with query parameters id=32196, convertId=2781 (USD), and range=START_UNIX~END_UNIX, where START_UNIX is the Unix timestamp (seconds) for 2025-09-26 00:00:00 UTC - 1 day and END_UNIX is the Unix timestamp (seconds) for 2025-09-26 00:00:00 UTC. Treat the interval as half-open: include START_UNIX and exclude END_UNIX. From the JSON response, read data.points (a mapping from timestamps to samples). For each entry (ts, point): if ts > 10_000_000_000 then ts is in milliseconds; divide by 1000 to obtain seconds. Extract the USD price as point.v[0] if point.v exists; otherwise use point.c.Discard any points with missing/NaN or non-positive prices. Consider only points with timestamps t satisfying START_UNIX ≤ t < END_UNIX, all in UTC. Compute the median of these USD prices. Multiply by 100, then report the number as an integer, rounded up.

</details>

## Market Structure
Each market contains **UP** and **DOWN** tokens representing directional bets on changes in the underlying metric:
- **UP tokens** increase in value when the metric rises above its starting level
- **DOWN tokens** increase in value when the metric falls below its starting level
- Token prices are obtained by a piece-wise linear mapping of the metric into a bounded range; the UP price is the scaled value and the DOWN price is `1 − UP`
- UP and DOWN prices always sum to **$1.00**, forming a complementary pair

## Price Mapping
- Market bounds: **min = 25**, **max = 85**. UP's USD price p is a linear mapping of the metric m into [0,1].
- Mapping: we scale the metric between min and max to get a number p between 0 and 1 (values below min map to 0; above max map to 1)
- DOWN's USD price is 1 − p
- AMM pool price (UP:DOWN) = p / (1 − p)
- Intuition: when the metric rises (p ↑), UP becomes more valuable relative to DOWN, so the pool price p/(1−p) increases; the LP's inventory rebalances toward DOWN and vice versa
- Impermanent loss depends on how far the pool price moves away from the starting price at your deposit; larger moves ⇒ larger IL (fees excluded here)

### Worked Example (for intuition)
- Take m at 60% of range: m = min + 0.60 × (max − min) = 61
- UP price: p = (m − min) / (max − min) = (61 − 25) / (85 − 25) = **0.600**
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
- **Mean** -11.28% and **median** -11.24% IL-only returns are shown below.
- These IL losses must be compared to incentive APY to calculate your net returns.

## Portfolio Performance
### IL Distribution Histogram
![Portfolio Return Distributions](il_hist.png)

### IL Over Time
![IL Returns Over Time](il_timeseries.png)

This time series shows how IL-only portfolio returns have varied across different historical windows.

### Distribution Summary (IL-only, %)

- Count: **14**
- Mean: **-11.28%**, Std: **2.14%**
- Median: **-11.24%**  |  P25: **-12.05%**  |  P10: **-13.99%**  |  P75: **-9.38%**

## Calculating Your Net APY

To determine your actual returns, combine Merkl incentive APY with these IL losses:

**Period Factor**: 0.066 (since this is a 24-day market)

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
**Example calculation only**: Let's say Merkl shows **200% APY** (this is just an example - actual APY varies by market) and you experience the **median IL loss (-11.24%)**:

1. **Scale Merkl APY to period**: 200% × 0.066 = 13.2%
2. **Convert to multiplier**: 1 + 13.2% = 1.132
3. **Apply median IL loss**: 1.132 × (1 + -11.2%) = 1.132 × 0.888 = 1.004
4. **Net return for 24 days**: 0.4%
5. **Annualized (APY)**: (1.004)^15.2 - 1 = **6.8% APY**

**Steps to use this with your actual numbers:**
1. Find your market's Merkl campaign and note the **actual APY** (not the 200% example)
   - **Note**: Merkl APY can vary over the duration of the market depending on the amount of liquidity provided
2. Multiply that APY by **0.066**
3. Add 1 to get the incentive multiplier
4. Multiply by (1 + your_expected_IL_return)
5. Subtract 1 to get your net return over 24 days
6. To annualize: raise (1 + return) to the power of 15.2, then subtract 1

# Technical Implementation

## Outputs
- **CSV (per-window IL)**: [il_by_window.csv](il_by_window.csv)

# Disclaimer
This analysis is for informational purposes only and does not constitute financial advice. Results are based on historical data and may not reflect future performance. Simulation code and models may contain errors or inaccuracies.