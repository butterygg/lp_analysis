#!/usr/bin/env python3
"""
Volume vs (Liquidity × Volatility) + Sampled Monthly IL%/$ for Uniswap Pools (v2/v3/v4)

Key features
- Per-subgraph schema autodetection (handles v4 variants that use currency0/currency1)
- Day data aggregation with robust fallbacks; reconstructs volumeUSD via tokenDayDatas when needed
- Monthly impermanent loss (IL) computed by short-horizon sampling & time-normalization (fees excluded)
- Saves: pool_metrics_daily.(csv|parquet), pool_metrics_monthly_il.csv, summary.json, scatter.png
"""

import os
import json
import time
import hashlib
import pickle
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm


# ---------------------------- Config ----------------------------


@dataclass
class Config:
    # scope
    N_TOP: int = 30
    LOOKBACK_DAYS: int = 60
    CHAINS: List[str] = None

    # volatility window
    VOL_WINDOW: int = 7
    MIN_VOLATILITY: float = 1e-6

    # depth approximation (until tick-walk is added)
    USE_FAST_APPROXIMATION: bool = True

    # IL sampling knobs (variance reduction), fees are excluded globally
    IL_SAMPLE_HORIZON_DAYS: int = 1  # h
    IL_SAMPLE_STEP_DAYS: int = 1  # sampling step
    IL_MIN_SAMPLES_PER_MONTH: int = 5
    IL_FALLBACK_TO_ENDPOINT: bool = True

    OUTPUT_DIR: str = "output"
    TRIM_PERCENTILE: float = 0.01

    def __post_init__(self):
        if self.CHAINS is None:
            self.CHAINS = ["mainnet", "base"]


# Subgraph IDs (update as needed)
SUBGRAPH_IDS = {
    "mainnet": {
        "v2": "A3Np3RQbaBA6oKJgiwDJeo5T3zrYfGHPWFYayMwtNDum",
        "v3": "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV",
        # "v4": "6XvRX3WHSvzBVTiPdF66XSBVbxWuHqijWANbjJxRDyzr",
    },
    "base": {
        # "v2": "4jGhpKjW4prWoyt5Bwk1ZHUwdEmNWveJcjEyjoTZWCY9",
        "v3": "43Hwfi3dJSoGpyas9VwNoDAv55yjgGrPpNSmbQZArzMG",
        # "v4": "2L6yxqUZ7dT6GWoTy9qxNBkf9kEk65me3XPMvbGsmJUZ",
    },
}


def get_subgraph_url(subgraph_id: str) -> str:
    return f"https://gateway.thegraph.com/api/subgraphs/id/{subgraph_id}"


# ---------------------------- Graph client ----------------------------


class SubgraphClient:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _headers(self):
        api_key = os.getenv("THE_GRAPH_API_KEY")
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    def query(
        self, endpoint: str, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        payload = {"query": query, "variables": variables or {}}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    endpoint, json=payload, headers=self._headers(), timeout=60
                )
                r.raise_for_status()
                data = r.json()
                if "errors" in data:
                    raise RuntimeError(data["errors"])
                return data.get("data", {})
            except Exception as e:
                last_err = e
                logger.warning(f"GraphQL attempt {attempt+1} failed: {e}")
                time.sleep(self.retry_delay * (2**attempt))
        raise RuntimeError(
            f"GraphQL failed after {self.max_retries} attempts: {last_err}"
        )

    def introspect_fields(self, endpoint: str, type_name: str) -> List[str]:
        q = """
        query __introspect($name: String!) {
          __type(name: $name) { fields { name } }
        }
        """
        try:
            data = self.query(endpoint, q, {"name": type_name})
            fields = data.get("__type", {}).get("fields", []) or []
            return [f["name"] for f in fields if "name" in f]
        except Exception as e:
            logger.warning(f"Introspection for {type_name} failed on {endpoint}: {e}")
            return []


# ---------------------------- Schema adapters ----------------------------


@dataclass
class PoolSchema:
    # names on the Pool type (some may be None/absent)
    token0_field: Optional[str]
    token1_field: Optional[str]
    fee_tier_field: Optional[str]
    liquidity_field: Optional[str]
    volume_usd_field: Optional[str]
    tvl_usd_field: Optional[str]
    # pool collection field
    pools_field: str = "pools"

    def has_minimal_tokens(self) -> bool:
        return (self.token0_field is not None) or (self.token1_field is not None)


def detect_pool_schema(
    client: SubgraphClient, endpoint: str, version: str
) -> PoolSchema:
    """Detect v2/v3/v4 Pool-ish fields present on this endpoint."""
    # v2 doesn't have Pool; it has Pair
    if version == "v2":
        return PoolSchema(None, None, None, None, None, None, pools_field="pairs")

    # try v3/v4 Pool
    pool_fields = set(client.introspect_fields(endpoint, "Pool"))
    # Some deployments expose Token fields as token0/token1 (classic),
    # some use currency0/currency1 (observed in explorer example for your v4 Base ID).
    token0 = None
    token1 = None
    if "token0" in pool_fields or "token1" in pool_fields:
        token0 = "token0" if "token0" in pool_fields else None
        token1 = "token1" if "token1" in pool_fields else None
    elif "currency0" in pool_fields or "currency1" in pool_fields:
        token0 = "currency0" if "currency0" in pool_fields else None
        token1 = "currency1" if "currency1" in pool_fields else None

    fee_tier = "feeTier" if "feeTier" in pool_fields else None
    liq = "liquidity" if "liquidity" in pool_fields else None
    vol_usd = "volumeUSD" if "volumeUSD" in pool_fields else None
    tvl_usd = "totalValueLockedUSD" if "totalValueLockedUSD" in pool_fields else None

    return PoolSchema(
        token0, token1, fee_tier, liq, vol_usd, tvl_usd, pools_field="pools"
    )


# ---------------------------- Query builders ----------------------------


def build_top_pools_query(schema: PoolSchema, first: int) -> Tuple[str, Dict[str, Any]]:
    """
    Build a safe 'top pools' query based on detected fields.
    Strategy:
      - Prefer orderBy: volumeUSD if available; else liquidity; else omit orderBy.
      - Request only fields that exist.
    """
    order_by = None
    if schema.volume_usd_field:
        order_by = schema.volume_usd_field
    elif schema.liquidity_field:
        order_by = schema.liquidity_field

    fields = ["id"]
    if schema.token0_field:
        fields.append(f"""{schema.token0_field} {{ id symbol decimals }}""")
    if schema.token1_field:
        fields.append(f"""{schema.token1_field} {{ id symbol decimals }}""")
    if schema.fee_tier_field:
        fields.append(schema.fee_tier_field)
    if schema.liquidity_field:
        fields.append(schema.liquidity_field)
    if schema.volume_usd_field:
        fields.append(schema.volume_usd_field)
    if schema.tvl_usd_field:
        fields.append(schema.tvl_usd_field)

    fields_block = "\n        ".join(fields)

    if order_by:
        q = f"""
        query GetTopPools($first: Int!) {{
          {schema.pools_field}(
            first: $first
            orderBy: {order_by}
            orderDirection: desc
          ) {{
            {fields_block}
          }}
        }}
        """
    else:
        q = f"""
        query GetTopPools($first: Int!) {{
          {schema.pools_field}(
            first: $first
          ) {{
            {fields_block}
          }}
        }}
        """
    return q, {"first": first}


Q_TOP_PAIRS_V2 = """
query GetTopPairs($first: Int!) {
  pairs(
    first: $first
    orderBy: volumeUSD
    orderDirection: desc
    where: { volumeUSD_gt: "1000" }
  ) {
    id
    token0 { id symbol decimals }
    token1 { id symbol decimals }
    reserveUSD
    volumeUSD
  }
}
"""

# Day data: unified superset; missing fields will be coerced to NaN later.
Q_POOL_DAYDATA_VX = """
query GetPoolDayData($poolId: String!, $since: Int!) {
  poolDayDatas(
    first: 1000
    orderBy: date
    orderDirection: asc
    where: { pool: $poolId, date_gt: $since }
  ) {
    date
    liquidity
    sqrtPrice
    token0Price
    token1Price
    volumeToken0
    volumeToken1
    volumeUSD
    tvlUSD
  }
}
"""

Q_PAIR_DAYDATA_V2 = """
query GetPairDayData($pairAddress: String!, $since: Int!) {
  pairDayDatas(
    first: 1000
    orderBy: date
    orderDirection: asc
    where: { pairAddress: $pairAddress, date_gt: $since }
  ) {
    date
    dailyVolumeUSD
    reserveUSD
    dailyVolumeToken0
    dailyVolumeToken1
  }
}
"""

Q_TOKEN_DAYDATA = """
query GetTokenDayData($tokenId: String!, $since: Int!) {
  tokenDayDatas(
    first: 1000
    orderBy: date
    orderDirection: asc
    where: { token: $tokenId, date_gt: $since }
  ) {
    date
    priceUSD
  }
}
"""


# ---------------------------- Analyzer ----------------------------


class UniswapAnalyzer:
    # Unified schema for calculate_metrics output and downstream consumers
    METRICS_COLS = [
        "date",
        "chain",
        "version",
        "pool_address",
        "token0",
        "token1",
        "feeTier",
        "y_volume_usd",
        "liquidity_depth_usd",
        "vol_pct",
        "x_liquidity_times_vol",
        "ratio_volume_over_liquidity_times_vol",
    ]

    def __init__(self, config: Config):
        self.config = config
        self.client = SubgraphClient()
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    # ---- Discovery ----

    def discover_top_pools(self) -> List[Dict[str, Any]]:
        logger.info("Discovering top pools across chains/versions…")
        candidates = []
        for chain in self.config.CHAINS:
            for version in self._get_versions_for_chain(chain):
                endpoint = self._get_subgraph_url(chain, version)
                if not endpoint:
                    continue
                try:
                    if version == "v2":
                        data = self.client.query(
                            endpoint, Q_TOP_PAIRS_V2, {"first": self.config.N_TOP * 3}
                        )
                        for p in data.get("pairs", []):
                            candidates.append(
                                {
                                    "id": p["id"],
                                    "chain": chain,
                                    "version": version,
                                    "token0": p["token0"],
                                    "token1": p["token1"],
                                    "feeTier": 3000,  # unused; kept for uniformity
                                    "volumeUSD": float(p.get("volumeUSD") or 0),
                                    "tvlUSD": float(p.get("reserveUSD") or 0),
                                }
                            )
                    else:
                        schema = detect_pool_schema(self.client, endpoint, version)
                        q, vars_ = build_top_pools_query(schema, self.config.N_TOP * 3)
                        data = self.client.query(endpoint, q, vars_)
                        pools = data.get(schema.pools_field, [])
                        for pool in pools:
                            # Tokens may be under token0/1 or currency0/1; both branches return dicts.
                            # Normalize symbols when available.
                            t0 = (
                                pool.get(schema.token0_field)
                                if schema.token0_field
                                else None
                            )
                            t1 = (
                                pool.get(schema.token1_field)
                                if schema.token1_field
                                else None
                            )
                            candidates.append(
                                {
                                    "id": pool["id"],
                                    "chain": chain,
                                    "version": version,
                                    "token0": (
                                        t0
                                        or {
                                            "id": None,
                                            "symbol": "NA",
                                            "decimals": None,
                                        }
                                    ),
                                    "token1": (
                                        t1
                                        or {
                                            "id": None,
                                            "symbol": "NA",
                                            "decimals": None,
                                        }
                                    ),
                                    "feeTier": (
                                        int(pool.get(schema.fee_tier_field) or 0)
                                        if schema.fee_tier_field
                                        else 0
                                    ),
                                    "volumeUSD": (
                                        float(pool.get(schema.volume_usd_field) or 0)
                                        if schema.volume_usd_field
                                        else 0.0
                                    ),
                                    "tvlUSD": (
                                        float(pool.get(schema.tvl_usd_field) or 0)
                                        if schema.tvl_usd_field
                                        else np.nan
                                    ),
                                }
                            )
                except Exception as e:
                    logger.error(f"Top pools failed on {chain} {version}: {e}")
        logger.info(f"Total candidate pools: {len(candidates)}")
        return candidates

    def _get_versions_for_chain(self, chain: str) -> List[str]:
        return list(SUBGRAPH_IDS.get(chain, {}).keys())

    def _get_subgraph_url(self, chain: str, version: str) -> Optional[str]:
        subgraph_id = SUBGRAPH_IDS.get(chain, {}).get(version)
        return get_subgraph_url(subgraph_id) if subgraph_id else None

    def _generate_cache_key(
        self, pool_id: str, chain: str, version: str, since_ts: int
    ) -> str:
        """Generate a cache key for pool daily data"""
        key_data = f"{pool_id}_{chain}_{version}_{since_ts}_{self.config.LOOKBACK_DAYS}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _load_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load cached pool data if it exists and is recent"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            return None

        try:
            # Check if cache is recent (within 1 hour)
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > 3600:  # 1 hour
                return None

            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None

    def _save_cached_data(self, cache_key: str, data: pd.DataFrame):
        """Save pool data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    # ---- Day data fetching ----

    def fetch_pool_daily_data(self, pools: List[Dict[str, Any]]) -> pd.DataFrame:
        logger.info("Fetching daily data…")
        since_ts = int(
            (
                datetime.now(timezone.utc) - timedelta(days=self.config.LOOKBACK_DAYS)
            ).timestamp()
        )
        all_parts: List[pd.DataFrame] = []

        # small cache for token day prices
        token_price_cache: Dict[Tuple[str, int], pd.DataFrame] = {}

        def get_token_prices(
            endpoint: str, token_id: Optional[str]
        ) -> Optional[pd.DataFrame]:
            if not token_id:
                return None
            key = (endpoint, hash(token_id))
            if key in token_price_cache:
                return token_price_cache[key]
            try:
                data = self.client.query(
                    endpoint, Q_TOKEN_DAYDATA, {"tokenId": token_id, "since": since_ts}
                )
                rows = data.get("tokenDayDatas", [])
                if not rows:
                    token_price_cache[key] = None
                    return None
                dfp = pd.DataFrame(rows)
                dfp["date"] = pd.to_datetime(dfp["date"], unit="s", utc=True).dt.floor(
                    "D"
                )
                dfp["priceUSD"] = pd.to_numeric(dfp["priceUSD"], errors="coerce")
                token_price_cache[key] = dfp[["date", "priceUSD"]]
                return token_price_cache[key]
            except Exception as e:
                logger.warning(f"Token price fetch failed for {token_id}: {e}")
                token_price_cache[key] = None
                return None

        for i, pool in enumerate(tqdm(pools, desc="Fetching pool daily data")):
            endpoint = self._get_subgraph_url(pool["chain"], pool["version"])
            if not endpoint:
                continue

            # Check cache first
            cache_key = self._generate_cache_key(
                pool["id"], pool["chain"], pool["version"], since_ts
            )
            cached_data = self._load_cached_data(cache_key)
            if cached_data is not None:
                logger.debug(f"Using cached data for pool {pool['id']}")
                all_parts.append(cached_data)
                continue

            try:
                if pool["version"] == "v2":
                    day = self.client.query(
                        endpoint,
                        Q_PAIR_DAYDATA_V2,
                        {"pairAddress": pool["id"], "since": since_ts},
                    )
                    rows = day.get("pairDayDatas", [])
                    if not rows:
                        continue
                    df = pd.DataFrame(rows)
                    df["date"] = pd.to_datetime(
                        df["date"], unit="s", utc=True
                    ).dt.floor("D")
                    df["volumeUSD"] = pd.to_numeric(
                        df["dailyVolumeUSD"], errors="coerce"
                    )
                    df["tvlUSD"] = pd.to_numeric(df["reserveUSD"], errors="coerce")
                    # price proxy via daily volumes if both tokens nonzero
                    if (
                        "dailyVolumeToken0" in df.columns
                        and "dailyVolumeToken1" in df.columns
                    ):
                        v0 = pd.to_numeric(df["dailyVolumeToken0"], errors="coerce")
                        v1 = pd.to_numeric(df["dailyVolumeToken1"], errors="coerce")
                        with np.errstate(divide="ignore", invalid="ignore"):
                            df["token0Price"] = (v1 / v0).replace(
                                [np.inf, -np.inf], np.nan
                            )
                    else:
                        df["token0Price"] = np.nan

                else:
                    # v3/v4
                    day = self.client.query(
                        endpoint,
                        Q_POOL_DAYDATA_VX,
                        {"poolId": pool["id"], "since": since_ts},
                    )
                    rows = day.get("poolDayDatas", [])
                    if not rows:
                        continue
                    df = pd.DataFrame(rows)
                    df["date"] = pd.to_datetime(
                        df["date"], unit="s", utc=True
                    ).dt.floor("D")
                    for col in [
                        "liquidity",
                        "sqrtPrice",
                        "token0Price",
                        "token1Price",
                        "volumeToken0",
                        "volumeToken1",
                        "volumeUSD",
                        "tvlUSD",
                    ]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                    # rebuild volumeUSD if absent
                    if ("volumeUSD" not in df.columns) or df["volumeUSD"].isna().all():
                        t0_id = pool["token0"].get("id")
                        t1_id = pool["token1"].get("id")
                        px0 = get_token_prices(endpoint, t0_id)
                        px1 = get_token_prices(endpoint, t1_id)
                        if (
                            px0 is not None
                            and px1 is not None
                            and "volumeToken0" in df.columns
                            and "volumeToken1" in df.columns
                        ):
                            m = df.merge(
                                px0, on="date", how="left", suffixes=("", "_t0")
                            )
                            m = m.rename(columns={"priceUSD": "priceUSD_t0"})
                            m = m.merge(
                                px1, on="date", how="left", suffixes=("", "_t1")
                            )
                            m = m.rename(columns={"priceUSD": "priceUSD_t1"})
                            df["volumeUSD"] = (
                                m["volumeToken0"] * m["priceUSD_t0"]
                                + m["volumeToken1"] * m["priceUSD_t1"]
                            )

                # metadata
                df["pool_id"] = pool["id"]
                df["chain"] = pool["chain"]
                df["version"] = pool["version"]
                df["token0_symbol"] = pool["token0"].get("symbol", "NA")
                df["token1_symbol"] = pool["token1"].get("symbol", "NA")
                df["feeTier"] = pool.get("feeTier", 0)

                # Cache the processed data
                self._save_cached_data(cache_key, df)
                all_parts.append(df)

            except Exception as e:
                logger.error(
                    f"Daily data failed for {pool['chain']} {pool['version']} {pool['id']}: {e}"
                )

        if not all_parts:
            raise RuntimeError("No daily data fetched")
        return pd.concat(all_parts, ignore_index=True)

    # ---- Daily metrics (volume, depth, vol) ----

    def _rolling_vol(self, prices: pd.Series, window: int) -> pd.Series:
        prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
        logret = np.log(prices / prices.shift(1))
        return logret.rolling(window=window).std()

    def _active_depth(self, pool_data: pd.DataFrame) -> pd.Series:
        # Until tick math: treat TVL as depth proxy if present, else NaN
        return pd.to_numeric(pool_data.get("tvlUSD"), errors="coerce")

    def _derive_midprice_from_sqrt(self, g: pd.DataFrame) -> Optional[pd.Series]:
        """
        If sqrtPrice (Q64.96) is present and usable, compute midprice = (sqrtPrice / 2**96)**2
        Returns a Series indexed by date or None.
        """
        if "sqrtPrice" not in g.columns or g["sqrtPrice"].notna().sum() < 3:
            return None
        q96 = g.set_index("date")["sqrtPrice"].astype(float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mid = (q96 / (2.0**96)) ** 2
        return mid.replace([np.inf, -np.inf], np.nan).dropna()

    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing daily metrics…")
        out = []

        # Define the output schema once so we can return an empty DF with columns if needed
        cols = list(self.METRICS_COLS)

        for pool_id, g in df.groupby("pool_id"):
            g = g.sort_values("date")
            meta = g.iloc[0]

            # ---- robust price selection (token0Price -> 1/token1Price -> sqrtPrice-derived) ----
            price_series = None
            if "token0Price" in g.columns and g["token0Price"].notna().sum() >= 3:
                price_series = g.set_index("date")["token0Price"].astype(float)
            elif "token1Price" in g.columns and g["token1Price"].notna().sum() >= 3:
                with np.errstate(divide="ignore", invalid="ignore"):
                    price_series = 1.0 / g.set_index("date")["token1Price"].astype(
                        float
                    )
            else:
                price_series = self._derive_midprice_from_sqrt(g)

            if price_series is None:
                continue

            price = price_series.replace([np.inf, -np.inf], np.nan).dropna()
            vol = self._rolling_vol(price, self.config.VOL_WINDOW)
            depth = self._active_depth(g)

            g2 = g.set_index("date").copy()
            g2["vol_pct"] = vol * 100.0
            g2["liquidity_depth_usd"] = depth
            g2["y_volume_usd"] = pd.to_numeric(g2.get("volumeUSD"), errors="coerce")
            g2["x_liquidity_times_vol"] = g2["liquidity_depth_usd"] * (
                g2["vol_pct"] / 100.0
            )
            g2 = g2.dropna(subset=["vol_pct", "liquidity_depth_usd", "y_volume_usd"])
            if g2.empty:
                continue

            g2 = g2.reset_index()
            g2["chain"] = meta["chain"]
            g2["version"] = meta["version"]
            g2["pool_address"] = pool_id
            g2["token0"] = meta["token0_symbol"]
            g2["token1"] = meta["token1_symbol"]
            with np.errstate(divide="ignore", invalid="ignore"):
                g2["ratio_volume_over_liquidity_times_vol"] = (
                    g2["y_volume_usd"] / g2["x_liquidity_times_vol"]
                )
            out.append(g2[cols])

        # Critical: preserve schema even when empty to avoid KeyError downstream
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)

    # ---- Monthly IL via sampling (fees excluded) ----

    @staticmethod
    def _il_percent(R: float) -> float:
        if R is None or not np.isfinite(R) or R <= 0:
            return np.nan
        return (2.0 * np.sqrt(R) / (1.0 + R)) - 1.0

    def _month_days(self, month_str: str) -> int:
        p = pd.Period(month_str, freq="M")
        return p.days_in_month

    def calculate_monthly_il_sampled(self, raw_daily: pd.DataFrame) -> pd.DataFrame:
        df = raw_daily.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["day"] = df["date"].dt.floor("D")
        df["month"] = df["date"].dt.to_period("M").astype(str)

        for c in [
            "token0Price",
            "token1Price",
            "tvlUSD",
            "reserveUSD",
            "volumeUSD",
            "sqrtPrice",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Build a robust daily price (prefer token0Price, else 1/token1Price, else sqrtPrice-derived)
        def robust_daily_price(g: pd.DataFrame) -> Optional[pd.Series]:
            if "token0Price" in g.columns and g["token0Price"].notna().sum() >= 2:
                s = (
                    g.dropna(subset=["token0Price"])
                    .set_index("day")["token0Price"]
                    .astype(float)
                )
                return s if not s.empty else None
            if "token1Price" in g.columns and g["token1Price"].notna().sum() >= 2:
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = 1.0 / g.dropna(subset=["token1Price"]).set_index("day")[
                        "token1Price"
                    ].astype(float)
                s = s.replace([np.inf, -np.inf], np.nan).dropna()
                return s if not s.empty else None
            # sqrtPrice path
            if "sqrtPrice" in g.columns and g["sqrtPrice"].notna().sum() >= 2:
                q = (
                    g.dropna(subset=["sqrtPrice"])
                    .set_index("day")["sqrtPrice"]
                    .astype(float)
                )
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    s = (q / (2.0**96)) ** 2
                s = s.replace([np.inf, -np.inf], np.nan).dropna()
                return s if not s.empty else None
            return None

        rows = []
        h = max(1, int(self.config.IL_SAMPLE_HORIZON_DAYS))
        step = max(1, int(self.config.IL_SAMPLE_STEP_DAYS))

        for (pool_id, month), g in df.groupby(["pool_id", "month"]):
            g = g.sort_values("day")
            version = g.iloc[0]["version"]
            chain = g.iloc[0]["chain"]
            token0 = g.iloc[0]["token0_symbol"]
            token1 = g.iloc[0]["token1_symbol"]
            feeTier = int(g.iloc[0]["feeTier"])

            tvl_series = g["reserveUSD"] if version == "v2" else g.get("tvlUSD")
            tvl_avg = (
                float(pd.to_numeric(tvl_series, errors="coerce").dropna().mean())
                if tvl_series is not None
                else np.nan
            )

            day_price = robust_daily_price(g)
            if day_price is None or day_price.empty:
                continue

            days = list(day_price.index.unique())
            days_set = set(days)
            samples = []
            idx = 0
            while idx < len(days):
                d0 = days[idx]
                d1 = d0 + pd.Timedelta(days=h)
                if d1 in days_set:
                    p0 = float(day_price.loc[d0])
                    p1 = float(day_price.loc[d1])
                    if np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0:
                        samples.append(self._il_percent(p1 / p0))
                idx += step

            W = self._month_days(month)
            if len(samples) >= self.config.IL_MIN_SAMPLES_PER_MONTH:
                il_per_h = float(np.nanmean(samples)) if len(samples) else np.nan
                il_month_pct = il_per_h * (W / h) if np.isfinite(il_per_h) else np.nan
                method = "sampled"
                n_samples = len(samples)
            else:
                il_month_pct = np.nan
                n_samples = len(samples)
                method = "insufficient_samples"
                if self.config.IL_FALLBACK_TO_ENDPOINT:
                    try:
                        p0 = float(day_price.dropna().iloc[0])
                        p1 = float(day_price.dropna().iloc[-1])
                        il_month_pct = (
                            self._il_percent(p1 / p0)
                            if p0 > 0 and np.isfinite(p0) and np.isfinite(p1)
                            else np.nan
                        )
                        method = "endpoint_fallback"
                    except Exception:
                        il_month_pct = np.nan

            il_month_usd = (
                il_month_pct * tvl_avg
                if (np.isfinite(il_month_pct) and np.isfinite(tvl_avg))
                else np.nan
            )
            rows.append(
                {
                    "pool_id": pool_id,
                    "chain": chain,
                    "version": version,
                    "token0": token0,
                    "token1": token1,
                    "feeTier": feeTier,
                    "month": month,
                    "calendar_days": W,
                    "il_method": method,
                    "n_il_samples": n_samples,
                    "il_pct_month": (
                        float(il_month_pct) if np.isfinite(il_month_pct) else np.nan
                    ),  # negative = loss
                    "tvl_usd_month_avg": (
                        float(tvl_avg) if np.isfinite(tvl_avg) else np.nan
                    ),
                    "il_usd_month": (
                        float(il_month_usd) if np.isfinite(il_month_usd) else np.nan
                    ),  # negative = loss $
                }
            )
        return pd.DataFrame(rows)

    # ---- Ranking / summary / viz ----

    def rank_and_filter_pools(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Ranking & filtering by 30d volume…")

        required = {"pool_address", "y_volume_usd", "liquidity_depth_usd", "vol_pct"}
        if df is None or df.empty or not required.issubset(df.columns):
            logger.warning("No valid daily metrics to rank; returning empty result.")
            return pd.DataFrame(columns=self.METRICS_COLS)

        pool_vols = (
            df.groupby("pool_address")["y_volume_usd"]
            .sum()
            .sort_values(ascending=False)
        )
        keep = set(pool_vols.head(self.config.N_TOP).index)
        out = df[df["pool_address"].isin(keep)].copy()
        out = out[
            (out["y_volume_usd"] > 0)
            & (out["liquidity_depth_usd"] > 0)
            & (out["vol_pct"] >= self.config.MIN_VOLATILITY * 100)
        ]
        logger.info(
            f"Final daily dataset: {len(out)} rows from {out['pool_address'].nunique()} pools"
        )
        return out

    def calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty:
            return {
                "number_of_pools": 0,
                "number_of_days": 0,
                "mean_ratio": np.nan,
                "median_ratio": np.nan,
                "trimmed_mean_ratio_1pct": np.nan,
                "correlation_logx_logy": np.nan,
                "ols_slope_log_log": np.nan,
                "analysis_date": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "lookback_days": self.config.LOOKBACK_DAYS,
                    "vol_window": self.config.VOL_WINDOW,
                    "il_sample_horizon_days": self.config.IL_SAMPLE_HORIZON_DAYS,
                    "il_sample_step_days": self.config.IL_SAMPLE_STEP_DAYS,
                    "il_min_samples_per_month": self.config.IL_MIN_SAMPLES_PER_MONTH,
                    "fees_included": False,
                },
            }

        ratios = (
            df["ratio_volume_over_liquidity_times_vol"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        def safe_q(s: pd.Series, q: float) -> float:
            try:
                return float(s.quantile(q))
            except Exception:
                return np.nan

        lower = safe_q(ratios, self.config.TRIM_PERCENTILE)
        upper = safe_q(ratios, 1 - self.config.TRIM_PERCENTILE)
        trimmed = (
            ratios
            if (np.isnan(lower) or np.isnan(upper))
            else ratios.clip(lower=lower, upper=upper)
        )

        x = np.log10(df["x_liquidity_times_vol"].replace(0, np.nan))
        y = np.log10(df["y_volume_usd"].replace(0, np.nan))
        common = x.dropna().index.intersection(y.dropna().index)
        corr = (
            np.corrcoef(x.loc[common], y.loc[common])[0, 1]
            if len(common) > 1
            else np.nan
        )
        slope = (
            np.polyfit(x.loc[common], y.loc[common], 1)[0]
            if len(common) > 1
            else np.nan
        )
        return {
            "number_of_pools": (
                int(df["pool_address"].nunique()) if "pool_address" in df.columns else 0
            ),
            "number_of_days": int(len(df)),
            "mean_ratio": float(ratios.mean()) if len(ratios) else np.nan,
            "median_ratio": float(ratios.median()) if len(ratios) else np.nan,
            "trimmed_mean_ratio_1pct": (
                float(trimmed.mean()) if len(trimmed) else np.nan
            ),
            "correlation_logx_logy": float(corr) if np.isfinite(corr) else np.nan,
            "ols_slope_log_log": float(slope) if np.isfinite(slope) else np.nan,
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "config": {
                "lookback_days": self.config.LOOKBACK_DAYS,
                "vol_window": self.config.VOL_WINDOW,
                "il_sample_horizon_days": self.config.IL_SAMPLE_HORIZON_DAYS,
                "il_sample_step_days": self.config.IL_SAMPLE_STEP_DAYS,
                "il_min_samples_per_month": self.config.IL_MIN_SAMPLES_PER_MONTH,
                "fees_included": False,
            },
        }

    def create_plots(self, df: pd.DataFrame, summary: Dict[str, Any]):
        logger.info("Rendering scatter plots…")
        if df is None or df.empty:
            logger.warning("No data to plot; creating empty scaffold figure.")
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No data available", ha="center", va="center")
            plt.axis("off")
            plt.savefig(self.output_dir / "scatter.png", dpi=300, bbox_inches="tight")
            plt.close()
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        colors = {"v2": "tab:red", "v3": "tab:blue", "v4": "tab:green"}

        for ver, g in df.groupby("version"):
            ax1.scatter(
                g["x_liquidity_times_vol"],
                g["y_volume_usd"],
                s=16,
                alpha=0.6,
                c=colors.get(ver, "gray"),
                label=f"{ver}",
            )
        ax1.set_xlabel("Liquidity Depth × Volatility (USD)")
        ax1.set_ylabel("Volume (USD)")
        ax1.set_title("Volume vs (Liquidity × Volatility)")
        ax1.legend()
        ax1.grid(alpha=0.3)

        for ver, g in df.groupby("version"):
            g2 = g[(g["x_liquidity_times_vol"] > 0) & (g["y_volume_usd"] > 0)]
            ax2.scatter(
                g2["x_liquidity_times_vol"],
                g2["y_volume_usd"],
                s=16,
                alpha=0.6,
                c=colors.get(ver, "gray"),
                label=f"{ver}",
            )
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Liquidity Depth × Volatility (USD)")
        ax2.set_ylabel("Volume (USD)")
        ax2.set_title("Log-Log")
        ax2.grid(alpha=0.3)
        if np.isfinite(summary.get("mean_ratio", np.nan)):
            xlim = ax2.get_xlim()
            ax2.plot(
                xlim,
                [x * summary["mean_ratio"] for x in xlim],
                "k--",
                alpha=0.5,
                label=f"Avg ratio = {summary['mean_ratio']:.2e}",
            )
            ax2.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

    def save_results(
        self, daily_df: pd.DataFrame, monthly_il: pd.DataFrame, summary: Dict[str, Any]
    ):
        self.output_dir.mkdir(exist_ok=True)
        daily_df.to_csv(self.output_dir / "pool_metrics_daily.csv", index=False)
        daily_df.to_parquet(self.output_dir / "pool_metrics_daily.parquet", index=False)
        monthly_il.to_csv(self.output_dir / "pool_metrics_monthly_il.csv", index=False)
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved to {self.output_dir}")

    # ---- Pipeline ----

    def run_analysis(self):
        logger.info("Starting analysis…")
        pools = self.discover_top_pools()
        raw_daily = self.fetch_pool_daily_data(pools)
        daily_metrics = self.calculate_metrics(raw_daily)
        final_daily = self.rank_and_filter_pools(daily_metrics)
        keep_ids = (
            set(final_daily["pool_address"].unique())
            if not final_daily.empty
            else set()
        )
        # If nothing passed the filters, monthly IL over keep_ids will be empty; still safe.
        monthly_il = self.calculate_monthly_il_sampled(
            raw_daily[raw_daily["pool_id"].isin(keep_ids)]
            if keep_ids
            else raw_daily.iloc[0:0]
        )
        summary = self.calculate_summary_stats(final_daily)
        self.create_plots(final_daily, summary)
        self.save_results(final_daily, monthly_il, summary)
        logger.info("Done.")


# ---------------------------- CLI ----------------------------


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Uniswap Volume vs Liquidity×Volatility + Sampled Monthly IL%/$ (fees excluded)"
    )
    p.add_argument("--n-top", type=int, default=30)
    p.add_argument("--lookback-days", type=int, default=30)
    p.add_argument("--chains", nargs="+", default=["mainnet", "base"])
    p.add_argument("--vol-window", type=int, default=7)
    p.add_argument("--il-sample-horizon-days", type=int, default=1)
    p.add_argument("--il-sample-step-days", type=int, default=1)
    p.add_argument("--il-min-samples-per-month", type=int, default=5)
    p.add_argument(
        "--no-endpoint-fallback",
        action="store_true",
        help="If set, months with insufficient samples yield NaN IL instead of endpoint fallback",
    )
    p.add_argument("--output-dir", default="output")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()

    config = Config(
        N_TOP=args.n_top,
        LOOKBACK_DAYS=args.lookback_days,
        CHAINS=args.chains,
        VOL_WINDOW=args.vol_window,
        OUTPUT_DIR=args.output_dir,
        USE_FAST_APPROXIMATION=args.fast,
        IL_SAMPLE_HORIZON_DAYS=args.il_sample_horizon_days,
        IL_SAMPLE_STEP_DAYS=args.il_sample_step_days,
        IL_MIN_SAMPLES_PER_MONTH=args.il_min_samples_per_month,
        IL_FALLBACK_TO_ENDPOINT=not args.no_endpoint_fallback,
    )
    UniswapAnalyzer(config).run_analysis()


if __name__ == "__main__":
    main()
