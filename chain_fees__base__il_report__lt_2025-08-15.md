# LP IL-Only Report — `chain_fees` / `base`

_Windows strictly earlier than **2025-08-15**. All figures exclude trading fees._

## Data Source
**Profile Question:** Use the DefiLlama fees API at https://api.llama.fi/summary/fees/base?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true&dataType=dailyRevenue.Return the average daily Base chain revenue in USD by computing the arithmetic mean of the per-day revenue values across all records with timestamps t satisfying{MARKET_START_DATE_UTC} < t ≤ {MARKET_END_DATE_UTC} (UTC).

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
- No valid points were available to compute a distribution for this cutoff.

## Portfolio Performance
### IL Distribution Histogram
_Histogram not available (no valid IL values to plot)._

### IL Over Time
_Time series graph not available (no valid IL values to plot)._

# Technical Implementation

## Outputs
- **CSV (per-window IL)**: [chain_fees__base__il_by_window__lt_2025-08-15.csv](chain_fees__base__il_by_window__lt_2025-08-15.csv)

# Disclaimer
This analysis is for informational purposes only and does not constitute financial advice. Results are based on historical data and may not reflect future performance. Simulation code and models may contain errors or inaccuracies.