# Spec — “Volume vs (Liquidity × Volatility)” for top Uniswap pools

## 0) Problem statement (what the script must deliver)

Over the most recent month (or the longest overlapping segment that data availability allows), for \~100 of the “top” Uniswap pools across v2/v3/v4:

1. For each **pool-day**:

   * Daily **volume in USD**.
   * Daily **liquidity depth in USD**.

     * v3/v4: use **active depth** around the current tick (configurable band).
     * v2: use full-range depth proxy (described below).
   * Daily **percent volatility** of the pool’s **USD price** (close-to-close).

2. Using these daily observations:

   * Build a scatter: **x = liquidity\_depth\_usd × vol\_pct**, **y = volume\_usd**.
   * Compute summary stats and the **average ratio** $\overline{\, V / (L \cdot \sigma)\,}$ across the sample, with and without trimming outliers and zero-vol cases.

3. Output:

   * A tidy CSV/Parquet of per-pool per-day metrics.
   * A PNG/HTML chart (scatter; optionally log–log), plus a small TXT/JSON with summary numbers.

---

## 1) Universe selection (which pools are the “top 100”?)

Make this a parameter; default to “top by 30-day USD volume.” There are two practical ways to get a good candidate set:

* **Direct via subgraphs** (preferred, avoids private APIs):

  * v3: query pools ordered by cumulative **volumeUSD** if available; otherwise, grab top by **liquidity** and then rank by 30-day volume computed from `poolDayDatas`. The Uniswap v3 subgraph exposes pool and daily aggregates, including `poolDayDatas { volumeToken0, volumeToken1, token0Price, token1Price }` which you can use to compute USD volumes day-by-day. ([Uniswap Docs][1])
  * v4: same pattern using the v4 subgraph’s `poolDayDatas` (fields are analogous: `volumeToken0`, `volumeToken1`, daily prices, etc.). ([Uniswap Docs][2])
  * v2: use `pairDayData` (volumeUSD, reserveUSD) to rank. ([Uniswap Docs][3])

* **Alternative (optional)**: seed from Uniswap App’s “Explore → Top Pools” then enrich via subgraphs. (Good UX reference; final data should still come from subgraphs to keep one source of truth.) ([app.uniswap.org][4])

Chains: parameterize (default `["mainnet", "arbitrum", "optimism", "base", "polygon"]`). Query per-chain subgraphs, then merge and rank globally. Uniswap documents versioned subgraphs and endpoints per chain. ([Uniswap Docs][5])

---

## 2) Data sources & fields (what to pull)

**Subgraphs (GraphQL):**

* **v3:** “v3 Protocol Query Examples” docs show how to fetch `poolDayDatas` (daily liquidity, sqrtPrice, token0Price, token1Price, volumes by token) and how to list/order pools. Ticks and liquidity state are accessible (for active depth). ([Uniswap Docs][1])

  * The **Active Liquidity** guide explains deriving the active liquidity at/around the current tick using tick data (`liquidityNet` per initialized tick). We’ll reuse that technique for an “active band.”
  * Liquidity math (convert between liquidity and token amounts within a price range): **LiquidityAmounts** library formulas are canonical. ([Uniswap Docs][6], [GitHub][7])

* **v4:** analogous examples exist (daily aggregates, tick fetching through StateView/multicall; pool object construction for off-chain math). ([Uniswap Docs][8])

* **v2:** the v2 subgraph exposes daily USD volume and reserve USD (`pairDayData`). ([Uniswap Docs][3])

**USD pricing:**

* For daily **volumeUSD** and **USD depth**, prefer subgraph-derived USD fields if present (e.g., v2 `pairDayData.volumeUSD`). For v3/v4, if `volumeUSD` isn’t in `poolDayDatas`, compute it as

  $$
  \text{volumeUSD}_t \approx \frac{ \text{volumeToken0}_t \cdot \text{price0USD}_t + \text{volumeToken1}_t \cdot \text{price1USD}_t }{2}
  $$

  where `priceUSD` per token per day can be pulled from `tokenDayDatas` (commonly includes `priceUSD`, OHLC), or derived via the subgraph “bundle” ETH price and token-to-WETH rates. (Both patterns are common in the Uniswap subgraph ecosystem.) ([Stack Overflow][9], [Uniswap Docs][3])

---

## 3) Metric definitions (precise, reproducible)

### 3.1 Daily volume in USD

* v2: `pairDayData.volumeUSD`.
* v3/v4: compute from `poolDayDatas.volumeToken0/1` × token USD prices as above, or use `poolDayDatas.volumeUSD` if your indexer exposes it.

### 3.2 Liquidity depth in USD (the “L” in $L \cdot \sigma$)

* **v3/v4 — Active depth “around current tick”:**
  Define a symmetric price band centered at current price:

  $$
  P_{\text{mid}} = P(\text{current tick}),\quad
  P_{\text{low}} = P_{\text{mid}}\,(1 - w),\quad
  P_{\text{high}} = P_{\text{mid}}\,(1 + w)
  $$

  where $w$ is a band half-width (default: **±1%**, configurable, or **±N ticks**).

  Procedure:

  1. Get **current tick** and **sqrtPriceX96** for the day’s close (use end-of-day `poolDayDatas.sqrtPrice` if available; otherwise nearest tick from swaps before UTC day end). ([Uniswap Docs][1])
  2. Convert $(P_{\text{low}}, P_{\text{high}}, P_{\text{mid}})$ into sqrt-prices and **ticks**.
  3. Traverse initialized ticks in $[ \text{tick}_{\text{low}}, \text{tick}_{\text{high}} ]$; at each tick interval, accumulate token0/token1 amounts **in-range** from the **cumulative active liquidity** curve. Use **LiquidityAmounts.getAmountsForLiquidity** on each sub-interval (clipped to your band) with the **active liquidity** for that segment. Sum token amounts over the band, then convert to USD at $P_{\text{mid}}$ (or daily token USD closes). This matches the Uniswap “Active Liquidity” approach and Uniswap’s periphery math. ([Uniswap Docs][6])
  4. **Active depth USD** is $\text{USD}( \text{sum of token0, token1 in band})$.

  Notes:

  * You’re effectively integrating liquidity density across the price band. The v3/v4 SDK docs and examples show how to fetch tick bitmaps and reconstruct tick liquidity efficiently. ([Uniswap Docs][8])
  * For robustness, cap the number of tick intervals visited per pool-day (e.g., stop if you pass 5,000 intervals) and fall back to a coarser band or to current-tick-only approximation (below).

  **Fast approximation (optional toggle):**
  Use **current-tick active liquidity only** and approximate a small-band depth by a first-order mapping between liquidity and amounts over $[P_{\text{low}}, P_{\text{high}}]$ via `LiquidityAmounts.getAmountsForLiquidity` **once** using the current in-range liquidity. This loses accuracy if liquidity density is highly skewed across the band but is 10–50× faster.

* **v2 — Full-range depth proxy:**
  v2 has no ticks; liquidity is uniform across $(0,\infty)$. For our “depth” proxy, use **reserveUSD** at day-end (`pairDayData.reserveUSD`), which is the USD value of both sides together. This matches the intuitive notion of “how much inventory is currently posted” (small-trade depth). It will overstate near-price “active” depth relative to v3, but that’s inherent to v2’s design and is acceptable if you flag it in the output. ([Uniswap Docs][3])

### 3.3 Daily volatility (the “σ” in $L \cdot \sigma$)

* Let $P_t$ be the **USD price of token0 in the pool** at end of day $t$.

  * v3/v4: `poolDayDatas.token0Price` is token0 in terms of token1; multiply by token1’s USD close, or compute USD price from token day data. ([Uniswap Docs][1])
  * v2: either use `pairDayData` to back out price (reserves ratio) times USD token price, or directly use token day prices if exposed.
* Daily log return $r_t = \ln(P_t/P_{t-1})$.
* **Percent volatility** $\sigma_t = 100 \times \text{stdev}\left(r_{t-h+1},\dots,r_t\right)$ using a **rolling window of 7 days** (configurable). If the month is short for a given pool, compute with what’s available; if fewer than 3 returns exist, mark as missing.
* To avoid divisions by \~0 in the ratio, clamp very low vol to an epsilon (e.g., $10^{-6}$) or exclude **stable-stable** pools from the ratio aggregation.

---

## 4) Calculations to produce the graph and the ratio

For every pool-day with all three metrics present:

* **xᵢ = L\_active\_USDᵢ × σᵢ**
* **yᵢ = volume\_USDᵢ**
* **ratioᵢ = yᵢ / xᵢ**

Then compute:

* Cross-sectional **average ratio** (mean/median of all ratioᵢ).
* The same after **trimming** (e.g., winsorize 1% tails) and **filtering** out zero-vol or zero-depth days.
* Optionally time-aggregate per pool (average over the month) before the cross-sectional average; report both cross-sectional and time-pooled means.

Plot:

* Scatter with **x on the horizontal**, **y on the vertical**. Provide a **log–log** toggle (recommended: log10 for both axes because both variables span orders of magnitude). Add optional OLS line and $R^2$.

---

## 5) Algorithm outline (pseudo)

```text
params:
  N_top = 100
  lookback_days = 30
  chains = ["mainnet","arbitrum","optimism","base","polygon"]
  band_type = "pct" | "ticks"
  band_width_pct = 0.01   # ±1%
  band_width_ticks = null
  vol_window = 7

# 1) Discover candidate pools per chain/version
candidates = []
for chain in chains:
  for version in [v3, v4, v2]:
    pools = top_pools(chain, version, N=3*N_top)   # oversample; rank later
    candidates.extend(pools)

# 2) Pull daily series for lookback_days
for pool in unique(candidates):
  daily = fetch_pool_daily(pool, lookback_days)
  prices = derive_usd_prices(pool, daily)  # token USD day closes (see §3.3)
  volumes = derive_usd_volumes(pool, daily, prices)
  if pool.version in {v3, v4}:
     ticks = fetch_initialized_ticks(pool) (or day-end snapshot if available)
     depth = active_depth_usd_series(pool, daily, ticks, band_type, band_width)
  else:  # v2
     depth = reserve_usd_series_from_pairDayData(pool)
  vol = rolling_volatility(prices, window=vol_window)

  join per-day: [date, volume_usd, depth_usd, vol_pct]

# 3) Rank pools by 30d volume_usd; keep top N_top
ranked = rank_by_sum_volume(candidates_series); keep top N_top

# 4) Compute x=L*σ, y=V, ratio=V/(L*σ). Save rows with all metrics present.

# 5) Output CSV/Parquet; make scatter (log-log optional). Compute summary stats.
```

---

## 6) Concrete GraphQL snippets (you can paste these in as “starter” queries)

### v3 / v4 — daily aggregates for a specific pool

```graphql
# inputs: $poolId, $since (unix day boundary)
{
  poolDayDatas(
    first: 1000
    orderBy: date
    orderDirection: asc
    where:{ pool: $poolId, date_gt: $since }
  ) {
    date
    liquidity
    sqrtPrice
    token0Price
    token1Price
    volumeToken0
    volumeToken1
  }
}
```

This pattern is documented in Uniswap’s v3 “Protocol Query Examples”, and v4 has a parallel example set. ([Uniswap Docs][1])

### v3 — tick data for active-depth

Fetch tick liquidity (‘liquidityNet’) around a band of ticks bracketing the current tick; accumulate into in-range liquidity per sub-interval. The “Active Liquidity” doc and “Fetching Pool Data” guides show the exact approach. ([Uniswap Docs][10])

### v2 — pair daily

```graphql
{
  pairDayDatas(
    first: 1000
    orderBy: date
    orderDirection: asc
    where:{ pairAddress: $pair, date_gt: $since }
  ) {
    date
    reserveUSD
    dailyVolumeUSD
  }
}
```

v2 query patterns are in the Uniswap v2 API/Queries docs. ([Uniswap Docs][3])

---

## 7) Numerical details (what the math is doing)

* **Price/tick conversions:** use Uniswap’s tick math; SDK gives `tickCurrent` and sqrt price. v3/v4 SDK & docs cover building the Pool and ticks list off-chain. ([Uniswap Docs][11])
* **Amounts from liquidity over a price range:** use `LiquidityAmounts.getAmountsForLiquidity(sqrtP, sqrtPa, sqrtPb, L)` (and companions). This is the authoritative formula set. ([Uniswap Docs][6], [GitHub][7])
* **Volatility:** log-return stdev (percent) over a small rolling window; clamp or skip tiny-vol observations.

---

## 8) Output schema (CSV/Parquet)

One row per pool-day meeting completeness criteria:

```
date, chain, version, pool_address, token0, token1, feeTier,
volume_usd, liquidity_depth_usd, vol_pct,
x_liquidity_times_vol, y_volume_usd,
ratio_volume_over_liquidity_times_vol,
band_type, band_param, notes
```

Also write:

* `summary.json` with: number\_of\_pools, number\_of\_days, mean\_ratio, median\_ratio, trimmed\_mean\_ratio(1%), correlation(logx, logy), OLS slope (log–log).
* `scatter.png` (and optionally `scatter_log.png`).

---

## 9) Plotting (aesthetics & diagnostics)

* Scatter of **x = L×σ** vs **y = V**.
* Render both normal and log–log; the latter reveals power-law relationships if present.
* Add thin reference lines (e.g., y = k·x) to visually interpret your **average ratio**; add a best-fit line and report $R^2$.
* Color points by **version** (v2/v3/v4) and shape by **chain** to spot structure.

---

## 10) Edge cases & safeguards

* **Stable-stable pools**: volatility near zero → huge ratios. Exclude observations with $\sigma < 10^{-6}$ or report them separately.
* **Missing USD prices** for exotic tokens: fallback to **bundle ETH/USD** and token/ETH price path if `priceUSD` is missing; otherwise drop the day. (Uniswap subgraphs commonly expose ETH-USD bundles and token daily USD prices either directly or derivable.) ([Uniswap Docs][3], [Stack Overflow][9])
* **Subgraph lags**: if the most recent day is incomplete, drop it (date < “yesterday” UTC).
* **v4 tick fetching**: large pools may have many initialized ticks; use multicall batching (documented) and limit band width if runtime is high. ([Uniswap Docs][8])
* **Ranking bias**: if initial ranking can’t trust cumulative `volumeUSD`, first preselect by **recent liquidity** and **swap count**, then compute 7-day or 30-day volumes to finalize the top set.

---

## 11) Directory structure & dependencies

* `src/config.py` — chain endpoints, band params, lookback, etc.
* `src/subgraph.py` — minimal GraphQL client; chain/version routing.
* `src/universe.py` — pool discovery & ranking.
* `src/daily.py` — per-pool daily series (prices, volumes, v2 reserves).
* `src/ticks.py` — v3/v4 tick fetch + active-depth integrator (exact & fast modes).
* `src/metrics.py` — volatility, L×σ, ratios.
* `src/plot.py` — scatter and regression.
* `main.py` — CLI entrypoint.

Dependencies (Python): `requests` (or `gql`), `pandas`, `numpy`, `matplotlib` (and `scipy` if you want OLS), optional `tenacity` for retries.

---

## 12) Validation plan

* **Spot-check**: pick a known pool (e.g., WETH/USDC 0.05% on mainnet). Compare your:

  * day volumes vs Uniswap UI analytics.
  * active depth vs a manual tick-by-tick integration over a narrow band.
* **Sanity**: if you set band to **full range** on a v3 full-range position pool, the active-depth USD should approximate the pool’s TVL (within fees and valuation nuances).
* **Invariance check**: widening the band should monotonically increase active depth.

---

## 13) Optional alternative approach (simpler, coarser, much faster)

If you’re happy with a **proxy** for active depth and want a 10× simpler pipeline:

* Define **liquidity depth USD** as **TVL in USD** at the daily close (`tvlUSD` or `reserveUSD`).
* Compute **σ** from daily USD price as above.
* Everything else the same.

This sacrifices the “around current tick” requirement for v3/v4 but is computationally light and still gets you the **shape** of the $V$ vs $L \cdot \sigma$ relationship.

---

## 14) Why this design matches Uniswap data structures

* The **daily aggregates** and **tick modeling** are exactly how Uniswap’s own docs show these queries and off-chain computations (poolDayDatas, tick bitmaps, active liquidity). ([Uniswap Docs][1])
* The **liquidity math** is pulled from Uniswap’s periphery `LiquidityAmounts` reference and the v3 book; using those formulas avoids home-rolled mistakes. ([Uniswap Docs][6], [uniswapv3book.com][12])

---

## 15) Minimal “getting started” code stubs (function signatures)

```python
# subgraph.py
def query(chain:str, version:str, gql:str, variables:dict)->dict: ...

# universe.py
def discover_top_pools(chains, versions, lookback_days, N_top)->list[dict]: ...

# daily.py
def fetch_pool_daydata(pool, since_ts)->pd.DataFrame: ...
def derive_usd_prices(pool, day_df)->pd.Series: ...
def derive_usd_volumes(pool, day_df, price_series)->pd.Series: ...

# ticks.py (v3/v4)
def fetch_ticks(pool)->pd.DataFrame: ...  # columns: tick, liquidityNet
def active_depth_usd_series(pool, day_df, ticks, band)->pd.Series: ...
# fast path:
def active_depth_usd_approx(pool, day_df)->pd.Series: ...

# metrics.py
def rolling_volatility(price_usd, window:int=7)->pd.Series: ...
def compute_xy_and_ratio(df)->pd.DataFrame: ...

# plot.py
def scatter_xy(df, loglog=True, annotate=False, out="scatter.png"): ...
```

---

If you want, I can turn this into a runnable Python skeleton next.

[1]: https://docs.uniswap.org/api/subgraph/guides/v3-examples "v3 Protocol Query Examples | Uniswap"
[2]: https://docs.uniswap.org/sdk/v3/guides/advanced/active-liquidity "Active Liquidity | Uniswap"
[3]: https://docs.uniswap.org/contracts/v2/reference/API/queries?utm_source=chatgpt.com "Queries - Uniswap"
[4]: https://app.uniswap.org/explore/pools?lng=en-US&utm_source=chatgpt.com "Explore top pools on Ethereum on Uniswap"
[5]: https://docs.uniswap.org/api/subgraph/overview?utm_source=chatgpt.com "Overview | Uniswap"
[6]: https://docs.uniswap.org/contracts/v3/reference/periphery/libraries/LiquidityAmounts?utm_source=chatgpt.com "LiquidityAmounts - Uniswap"
[7]: https://github.com/Uniswap/v3-periphery/blob/main/contracts/libraries/LiquidityAmounts.sol?utm_source=chatgpt.com "v3-periphery/contracts/libraries/LiquidityAmounts.sol at main · Uniswap ..."
[8]: https://docs.uniswap.org/sdk/v4/guides/advanced/pool-data "Fetching Pool Data | Uniswap"
[9]: https://stackoverflow.com/questions/71730020/how-to-use-tokendaydata-in-uniswap?utm_source=chatgpt.com "How to use TokenDayData in Uniswap - Stack Overflow"
[10]: https://docs.uniswap.org/sdk/v3/guides/advanced/pool-data?utm_source=chatgpt.com "Fetching Pool Data - Uniswap"
[11]: https://docs.uniswap.org/sdk/v3/reference/classes/Pool?utm_source=chatgpt.com "Pool | Uniswap"
[12]: https://uniswapv3book.com/milestone_1/calculating-liquidity.html?utm_source=chatgpt.com "Calculating Liquidity - Uniswap V3 Development Book"