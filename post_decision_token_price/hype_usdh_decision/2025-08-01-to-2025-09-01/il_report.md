# LP IL-Only Report — `post_decision_token_price` / `hype_usdh_decision`

_Windows strictly earlier than **2025-08-01**. All figures exclude trading fees._

## Data Source
**Metric which market is forecasting:** HYPE price (USD) - daily median on Sep 1, 2025

<details><summary>Oracle Question used to resolve metric value</summary>

Use the CoinMarketCap detail/chart endpoint at https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail/chart with query parameters id=32196, convertId=2781 (USD), and range=START_UNIX~END_UNIX. Resolve DECISION_TIME_UTC from ${DISAMBIGUATION_URI} (the USDH validator-vote decision time, in UTC). Set START_UNIX to the Unix timestamp (seconds) at DECISION_TIME_UTC + 120 minutes, and set END_UNIX to START_UNIX + 43_200. Treat the interval as half-open: include START_UNIX and exclude END_UNIX. From the JSON response, read data.points (a mapping from timestamps to samples). For each entry (ts, point): if ts > 10_000_000_000 then ts is in milliseconds; divide by 1000 to obtain seconds. Extract the USD price as point.v[0] if point.v exists; otherwise use point.c. Discard any points with missing/NaN or non-positive prices. Consider only points with timestamps t satisfying START_UNIX ≤ t < END_UNIX (UTC). Compute the median of these USD prices. Multiply by 100, then report the number as an integer, rounded up.

</details>

## Decision Market Note
- Decision: Who should issue USDH?
- Decision date: 2025-09-14 11:00 UTC (2025-09-14T11:00:00Z)
- This is a decision market. The decision may or may not lead to additional volatility and impermanent loss relative to the modeled analysis here, depending on how unexpected the final decision is and how impactful it is.

## Market Structure
Each market contains **UP** and **DOWN** tokens representing directional bets on changes in the underlying metric:
- **UP tokens** increase in value when the metric rises above its starting level
- **DOWN tokens** increase in value when the metric falls below its starting level
- Token prices are obtained by a piece-wise linear mapping of the metric into a bounded range; the UP price is the scaled value and the DOWN price is `1 − UP`
- UP and DOWN prices always sum to **$1.00**, forming a complementary pair

## Price Mapping
- Market bounds: **min = 15**, **max = 95**. UP's USD price p is a linear mapping of the metric m into [0,1].
- Mapping: we scale the metric between min and max to get a number p between 0 and 1 (values below min map to 0; above max map to 1)
- DOWN's USD price is 1 − p
- AMM pool price (UP:DOWN) = p / (1 − p)
- Intuition: when the metric rises (p ↑), UP becomes more valuable relative to DOWN, so the pool price p/(1−p) increases; the LP's inventory rebalances toward DOWN and vice versa
- Impermanent loss depends on how far the pool price moves away from the starting price at your deposit; larger moves ⇒ larger IL (fees excluded here)

### Worked Example (for intuition)
- Take m at 60% of range: m = min + 0.60 × (max − min) = 63
- UP price: p = (m − min) / (max − min) = (63 − 15) / (95 − 15) = **0.600**
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
- No valid points were available to compute a distribution for this cutoff.

## Portfolio Performance
### IL Distribution Histogram
_Histogram not available (no valid IL values to plot)._

### IL Over Time
_Time series graph not available (no valid IL values to plot)._

# Technical Implementation

## Outputs
- **CSV (per-window IL)**: [il_by_window.csv](il_by_window.csv)

# Disclaimer
This analysis is for informational purposes only and does not constitute financial advice. Results are based on historical data and may not reflect future performance. Simulation code and models may contain errors or inaccuracies.