#!/usr/bin/env python3
"""
Daily Volume  ↔  Daily Impermanent-Loss (IL) for Uniswap pools (v2 / v3).

Pipeline: fetch → compute → analyse → plot → persist
Reporting metric:  il_per_1k_volume  (= |avg daily IL| / avg daily volume * 1000)

Key features:
- Timeseries-first discovery (default) for v3 AND v2: collect pools/pairs that actually have day/hour rows in the lookback window.
- Automatic fallback to entity-first discovery if timeseries-first yields no candidates.
- Optional entity-first discovery (lifetime volume ranking) with lifetime volume prefiltering.
- Excludes stable↔stable pools by filtering pools whose median price ≈ 1.
- Fallback for v3 pools with missing poolDayDatas: aggregate poolHourDatas → daily.
- Outlier removal using log-MAD on il_per_1k_volume.
- Boundary-inclusive filters (date_gte / periodStartUnix_gte).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Imports                                                                     #
# --------------------------------------------------------------------------- #
import os, json, time, pickle, hashlib, argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from loguru import logger
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Constants / Config                                                          #
# --------------------------------------------------------------------------- #
SUBGRAPH_IDS = {
    "mainnet": {
        "v2": "A3Np3RQbaBA6oKJgiwDJeo5T3zrYfGHPWFYayMwtNDum",
        "v3": "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV",
    },
    "base": {
        # NOTE: If this ever goes stale for day/hour entities, the v3 fallback will still work.
        "v3": "43Hwfi3dJSoGpyas9VwNoDAv55yjgGrPpNSmbQZArzMG",
    },
}


class Config:
    def __init__(
        self,
        n_top: int = 100,
        lookback_days: int = 60,
        chains: list[str] | None = None,
        cache_ttl: int = 60 * 60 * 24,
        output_dir: str = "output",
        # pools whose median price is within [1 - eps, 1 + eps] are treated as stable↔stable and excluded
        stable_price_eps: float = 0.03,
        # pools with extremely tiny avg |IL| (≈0) make ratios explode; drop them
        min_avg_abs_il_usd: float = 1e-6,
        # outlier removal strength (log-MAD fence multiplier)
        mad_k: float = 2.5,
        # discovery mode and lifetime volume prefilters (coarse, server-side)
        discovery_mode: str = "timeseries",  # "timeseries" or "entity"
        min_discovery_lifetime_volume_usd: float | None = None,
        max_discovery_lifetime_volume_usd: float | None = None,
        # exact analysis-time filters on computed avg daily volume over lookback
        min_avg_daily_volume_usd: float | None = None,
        max_avg_daily_volume_usd: float | None = None,
    ) -> None:
        self.N_TOP = n_top
        self.LOOKBACK_DAYS = lookback_days
        self.CHAINS = chains or ["mainnet", "base"]
        self.CACHE_TTL = cache_ttl
        self.OUTPUT_DIR = output_dir
        self.STABLE_PRICE_EPS = stable_price_eps
        self.MIN_AVG_ABS_IL_USD = min_avg_abs_il_usd
        self.MAD_K = mad_k

        self.DISCOVERY_MODE = discovery_mode
        self.MIN_DISCOVERY_LIFETIME_VOLUME_USD = min_discovery_lifetime_volume_usd
        self.MAX_DISCOVERY_LIFETIME_VOLUME_USD = max_discovery_lifetime_volume_usd
        self.MIN_AVG_DAILY_VOLUME_USD = min_avg_daily_volume_usd
        self.MAX_AVG_DAILY_VOLUME_USD = max_avg_daily_volume_usd


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def epoch_days_ago(days: int) -> int:
    today_utc = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int((today_utc - timedelta(days=days)).timestamp())


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def iqr_bounds(s: pd.Series, k: float = 3.0) -> Tuple[float, float]:
    s = s[np.isfinite(s)]
    if s.empty:
        return -np.inf, np.inf
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def mad_bounds_log(s: pd.Series, k: float = 2.5) -> Tuple[float, float]:
    """
    Robust bounds in LOG space for heavy-tailed positive data.
    Keeps values whose log10 lie within median ± k * 1.4826 * MAD.
    Returns bounds in the original (linear) space.
    """
    s = s[(s > 0) & np.isfinite(s)]
    if s.empty:
        # Domain-aware sentinel for positive variable
        return 0.0, np.inf
    log_s = np.log10(s)
    med = np.nanmedian(log_s)
    mad = np.nanmedian(np.abs(log_s - med))
    if not np.isfinite(med) or not np.isfinite(mad) or mad == 0:
        lo_q, hi_q = np.nanpercentile(s, [1, 99])
        return float(max(lo_q, 0.0)), float(hi_q)
    scale = 1.4826 * mad
    lo, hi = med - k * scale, med + k * scale
    return float(10**lo), float(10**hi)


# --------------------------------------------------------------------------- #
# GraphQL Client                                                              #
# --------------------------------------------------------------------------- #
class SubgraphClient:
    def __init__(self, retries: int = 3, delay: float = 1.0) -> None:
        self.retries, self.delay = retries, delay
        self.headers = {"Content-Type": "application/json"}
        if k := os.getenv("THE_GRAPH_API_KEY"):
            self.headers["Authorization"] = f"Bearer {k}"

    def query(self, url: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = {"query": query, "variables": variables}
        err: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = requests.post(url, json=payload, headers=self.headers, timeout=60)
                r.raise_for_status()
                data = r.json()
                if "errors" in data:
                    raise RuntimeError(data["errors"])
                return data["data"]
            except Exception as exc:
                err = exc
                logger.warning(f"GQL attempt {attempt+1}/{self.retries} failed: {exc}")
                time.sleep(self.delay * 2**attempt)
        raise RuntimeError(err)


# --------------------------------------------------------------------------- #
# Disk cache                                                                  #
# --------------------------------------------------------------------------- #
class DiskCache:
    def __init__(self, root: Path, ttl: int) -> None:
        self.root, self.ttl = root, ttl
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> Optional[pd.DataFrame]:
        p = self.root / f"{key}.pkl"
        if not p.exists() or (time.time() - p.stat().st_mtime) > self.ttl:
            return None
        try:
            return pickle.loads(p.read_bytes())
        except Exception:
            return None

    def save(self, key: str, df: pd.DataFrame) -> None:
        try:
            (self.root / f"{key}.pkl").write_bytes(pickle.dumps(df))
        except Exception as exc:
            logger.warning(f"Cache save failed for {key}: {exc}")


# --------------------------------------------------------------------------- #
# Core analyser                                                               #
# --------------------------------------------------------------------------- #
class UniswapILVolumeAnalyser:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = SubgraphClient()
        self.output = Path(cfg.OUTPUT_DIR)
        self.output.mkdir(exist_ok=True, parents=True)
        self.cache = DiskCache(self.output / "cache", cfg.CACHE_TTL)

    # ------------------------------ discovery ------------------------------ #
    def _url(self, chain: str, version: str) -> str | None:
        sg = SUBGRAPH_IDS.get(chain, {}).get(version)
        return f"https://gateway.thegraph.com/api/subgraphs/id/{sg}" if sg else None

    def discover_pools(self) -> list[dict[str, Any]]:
        if self.cfg.DISCOVERY_MODE == "timeseries":
            pools = self._discover_pools_timeseries_first()
            if not pools:
                logger.warning(
                    "Timeseries-first discovery returned 0 pools; falling back to entity-first."
                )
                pools = self._discover_pools_entity_first()
            return pools
        return self._discover_pools_entity_first()

    def _discover_pools_entity_first(self) -> list[dict[str, Any]]:
        """
        Lifetime-volume ranking, with optional server-side prefilters on lifetime volumeUSD.
        """
        pools: list[dict[str, Any]] = []
        for chain in self.cfg.CHAINS:
            for version in SUBGRAPH_IDS.get(chain, {}):
                url = self._url(chain, version)
                if not url:
                    continue
                try:
                    minv = self.cfg.MIN_DISCOVERY_LIFETIME_VOLUME_USD
                    maxv = self.cfg.MAX_DISCOVERY_LIFETIME_VOLUME_USD
                    if version == "v2":
                        where_parts, var_sig = [], []
                        vars: dict[str, Any] = {"n": self.cfg.N_TOP}
                        if minv is not None:
                            where_parts.append("volumeUSD_gte:$minVol")
                            var_sig.append("$minVol: BigDecimal")
                            vars["minVol"] = float(minv)
                        if maxv is not None:
                            where_parts.append("volumeUSD_lte:$maxVol")
                            var_sig.append("$maxVol: BigDecimal")
                            vars["maxVol"] = float(maxv)
                        where_str = (
                            f"where:{{{', '.join(where_parts)}}}" if where_parts else ""
                        )
                        vsig = (", " + ", ".join(var_sig)) if var_sig else ""
                        q = f"""
                        query($n:Int!{vsig}) {{
                          pairs(first:$n, orderBy:volumeUSD, orderDirection:desc {(','+where_str) if where_str else ''}) {{
                            id reserveUSD volumeUSD
                            token0{{id symbol}} token1{{id symbol}}
                          }}
                        }}"""
                        rows = self.client.query(url, q, vars).get("pairs", [])
                        for r in rows:
                            pools.append(
                                dict(
                                    id=r["id"],
                                    version=version,
                                    chain=chain,
                                    token0=r["token0"],
                                    token1=r["token1"],
                                )
                            )
                    else:  # v3
                        where_parts, var_sig = [], []
                        vars: dict[str, Any] = {"n": self.cfg.N_TOP}
                        if minv is not None:
                            where_parts.append("volumeUSD_gte:$minVol")
                            var_sig.append("$minVol: BigDecimal")
                            vars["minVol"] = float(minv)
                        if maxv is not None:
                            where_parts.append("volumeUSD_lte:$maxVol")
                            var_sig.append("$maxVol: BigDecimal")
                            vars["maxVol"] = float(maxv)
                        where_str = (
                            f"where:{{{', '.join(where_parts)}}}" if where_parts else ""
                        )
                        vsig = (", " + ", ".join(var_sig)) if var_sig else ""
                        q = f"""
                        query($n:Int!{vsig}) {{
                          pools(first:$n, orderBy:volumeUSD, orderDirection:desc {(','+where_str) if where_str else ''}) {{
                            id totalValueLockedUSD volumeUSD
                            token0{{id symbol}} token1{{id symbol}}
                          }}
                        }}"""
                        rows = self.client.query(url, q, vars).get("pools", [])
                        for r in rows:
                            pools.append(
                                dict(
                                    id=r["id"],
                                    version=version,
                                    chain=chain,
                                    token0=r["token0"],
                                    token1=r["token1"],
                                )
                            )
                except Exception as exc:
                    logger.error(f"top-pool query failed: {chain} {version}: {exc}")
        logger.info(f"Discovered {len(pools)} pools (entity-first)")
        return pools

    def _discover_pools_timeseries_first(self) -> list[dict[str, Any]]:
        """
        Timeseries-first discovery for BOTH v3 and v2.
        - v3: poolDayDatas / poolHourDatas (nested pool → token symbols available).
        - v2: pairDayDatas exposes pairAddress, token0, token1 directly.
        Downselect to N_TOP by true window volume (sum over day/hour rows).
        """
        since = epoch_days_ago(self.cfg.LOOKBACK_DAYS)
        target_unique = (
            self.cfg.N_TOP * 50
        )  # gather many more candidates for better low-volume pool selection
        pools_accum: dict[tuple[str, str], dict[str, Any]] = (
            {}
        )  # (chain,id) -> info + _window_volume

        for chain in self.cfg.CHAINS:
            for version in SUBGRAPH_IDS.get(chain, {}):
                url = self._url(chain, version)
                if not url:
                    continue

                if version == "v3":

                    def harvest_v3(query: str, key: str) -> None:
                        skip = 0
                        PAGE = 1000
                        while True:
                            vars = {"ts": since, "first": PAGE, "skip": skip}
                            data = self.client.query(url, query, vars)
                            rows = data.get(key, [])
                            if not rows:
                                break
                            for r in rows:
                                p = r["pool"]
                                pid = p["id"]
                                k = (chain, pid)
                                vol = float(r.get("volumeUSD") or 0.0)
                                if k not in pools_accum:
                                    pools_accum[k] = dict(
                                        id=pid,
                                        version=version,
                                        chain=chain,
                                        token0=p["token0"],
                                        token1=p["token1"],
                                        _window_volume=0.0,
                                    )
                                pools_accum[k]["_window_volume"] += vol
                            skip += len(rows)
                            if len(pools_accum) >= target_unique:
                                break

                    q_day = """
                    query($ts:Int!,$first:Int!,$skip:Int!){
                      poolDayDatas(first:$first, skip:$skip, orderBy:volumeUSD, orderDirection:asc,
                                   where:{date_gte:$ts}){
                        volumeUSD
                        pool{
                          id
                          token0{id symbol}
                          token1{id symbol}
                        }
                      }
                    }"""
                    harvest_v3(q_day, "poolDayDatas")

                    if len(pools_accum) < self.cfg.N_TOP:
                        q_hour = """
                        query($ts:Int!,$first:Int!,$skip:Int!){
                          poolHourDatas(first:$first, skip:$skip, orderBy:volumeUSD, orderDirection:asc,
                                        where:{periodStartUnix_gte:$ts}){
                            volumeUSD
                            pool{
                              id
                              token0{id symbol}
                              token1{id symbol}
                            }
                          }
                        }"""
                        harvest_v3(q_hour, "poolHourDatas")

                else:  # v2

                    def harvest_v2(query: str) -> None:
                        skip = 0
                        PAGE = 1000
                        while True:
                            vars = {"ts": since, "first": PAGE, "skip": skip}
                            data = self.client.query(url, query, vars)
                            rows = data.get("pairDayDatas", [])
                            if not rows:
                                break
                            for r in rows:
                                pid = r.get("pairAddress")
                                if not pid:
                                    # As a fallback, some deployments use id = "<pair>-<day>"
                                    pid = (r.get("id", "").split("-") or [""])[0]
                                if not pid:
                                    continue
                                k = (chain, pid)
                                vol = float(r.get("dailyVolumeUSD") or 0.0)
                                t0 = r.get("token0")
                                t1 = r.get("token1")
                                if k not in pools_accum:
                                    pools_accum[k] = dict(
                                        id=pid,
                                        version=version,
                                        chain=chain,
                                        token0=t0,
                                        token1=t1,
                                        _window_volume=0.0,
                                    )
                                else:
                                    if t0 and not pools_accum[k].get("token0"):
                                        pools_accum[k]["token0"] = t0
                                    if t1 and not pools_accum[k].get("token1"):
                                        pools_accum[k]["token1"] = t1
                                pools_accum[k]["_window_volume"] += vol
                            skip += len(rows)
                            if len(pools_accum) >= target_unique:
                                break

                    q_v2 = """
                    query($ts:Int!,$first:Int!,$skip:Int!){
                      pairDayDatas(first:$first, skip:$skip, orderBy:dailyVolumeUSD, orderDirection:asc,
                                   where:{date_gte:$ts}){
                        id
                        pairAddress
                        dailyVolumeUSD
                        token0{ id symbol }
                        token1{ id symbol }
                      }
                    }"""
                    harvest_v2(q_v2)

        items = sorted(
            pools_accum.values(), key=lambda x: x["_window_volume"], reverse=False
        )
        # Skip the very lowest volume pools (likely bad data) and take from a range
        # that balances low volume with data quality
        start_idx = len(items) // 20  # Skip bottom 5%
        end_idx = start_idx + self.cfg.N_TOP * 2  # Take 2x more than needed
        candidate_range = items[start_idx:end_idx]
        picked = candidate_range[: self.cfg.N_TOP]
        logger.info(
            f"Discovered {len(picked)} pools (timeseries-first) from {len(pools_accum)} candidates"
        )
        for it in picked:
            it.pop("_window_volume", None)
        return picked

    # ------------------------------ daily data ----------------------------- #
    def _aggregate_hourly_to_daily(self, dfh: pd.DataFrame) -> pd.DataFrame:
        """
        Convert poolHourDatas to a daily-like frame compatible with poolDayDatas.
        We use end-of-day (last observed) prices/tvl and sum volumes over the day.
        """
        if dfh.empty:
            return dfh

        for col in [
            "volumeUSD",
            "tvlUSD",
            "token0Price",
            "token1Price",
            "sqrtPrice",
            "periodStartUnix",
        ]:
            if col not in dfh:
                dfh[col] = np.nan
            dfh[col] = to_num(dfh[col])

        dfh["datetime"] = pd.to_datetime(dfh["periodStartUnix"], unit="s", utc=True)
        dfh = dfh.sort_values("datetime")
        dfh["date"] = dfh["datetime"].dt.floor("D")

        def last_non_null(s: pd.Series) -> float:
            s = s.dropna()
            return float(s.iloc[-1]) if not s.empty else float("nan")

        daily = dfh.groupby("date", as_index=False).agg(
            volumeUSD=("volumeUSD", "sum"),
            tvlUSD=("tvlUSD", last_non_null),
            token0Price=("token0Price", last_non_null),
            token1Price=("token1Price", last_non_null),
            sqrtPrice=("sqrtPrice", last_non_null),
        )
        return daily

    def fetch_daily(self, pools: list[dict[str, Any]]) -> pd.DataFrame:
        if not pools:
            raise RuntimeError("discovery returned no pools; nothing to fetch")
        since = epoch_days_ago(self.cfg.LOOKBACK_DAYS)
        frames: list[pd.DataFrame] = []
        skipped_missing_ts: list[tuple[str, str]] = []  # (chain, pool_id)

        for p in tqdm(pools, desc="fetch-daily"):
            url = self._url(p["chain"], p["version"])
            if not url:
                continue
            ck = md5(f"{p['id']}-{since}")
            if (df := self.cache.load(ck)) is not None:
                frames.append(df)
                continue

            try:
                if p["version"] == "v2":
                    q = """
                    query($addr:String!,$ts:Int!){
                      pairDayDatas(first:1000, orderBy:date, orderDirection:asc,
                        where:{pairAddress:$addr, date_gte:$ts}){
                          date
                          dailyVolumeUSD
                          reserveUSD
                          dailyVolumeToken0
                          dailyVolumeToken1
                    }}"""
                    rows = self.client.query(
                        url, q, {"addr": p["id"], "ts": since}
                    ).get("pairDayDatas", [])
                    df = pd.DataFrame(rows)

                    if df.empty:
                        skipped_missing_ts.append((p["chain"], p["id"]))
                        continue

                    for col in [
                        "dailyVolumeUSD",
                        "reserveUSD",
                        "dailyVolumeToken0",
                        "dailyVolumeToken1",
                    ]:
                        if col not in df:
                            df[col] = np.nan
                        df[col] = to_num(df[col])

                    if (
                        "dailyVolumeToken0" in df
                        and "dailyVolumeToken1" in df
                        and (
                            df["dailyVolumeToken0"].notna().any()
                            or df["dailyVolumeToken1"].notna().any()
                        )
                    ):
                        with np.errstate(divide="ignore", invalid="ignore"):
                            df["token0Price"] = (
                                df["dailyVolumeToken1"] / df["dailyVolumeToken0"]
                            )
                    else:
                        df["token0Price"] = np.nan

                    df["tvlUSD"] = df["reserveUSD"]
                    df["volumeUSD"] = df["dailyVolumeUSD"]

                else:  # v3
                    q_day = """
                    query($id:String!,$ts:Int!){
                      poolDayDatas(first:1000, orderBy:date, orderDirection:asc,
                        where:{pool:$id, date_gte:$ts}){
                          date
                          volumeUSD
                          tvlUSD
                          token0Price
                          token1Price
                          sqrtPrice
                    }}"""
                    response = self.client.query(
                        url, q_day, {"id": p["id"], "ts": since}
                    )
                    rows = response.get("poolDayDatas", [])
                    df = pd.DataFrame(rows)

                    if df.empty:
                        q_hour = """
                        query($id:String!,$ts:Int!){
                          poolHourDatas(first:1000, orderBy:periodStartUnix, orderDirection:asc,
                            where:{pool:$id, periodStartUnix_gte:$ts}){
                              periodStartUnix
                              volumeUSD
                              tvlUSD
                              token0Price
                              token1Price
                              sqrtPrice
                        }}"""
                        resp_h = self.client.query(
                            url, q_hour, {"id": p["id"], "ts": since}
                        )
                        rows_h = resp_h.get("poolHourDatas", [])
                        dfh = pd.DataFrame(rows_h)
                        if dfh.empty:
                            skipped_missing_ts.append((p["chain"], p["id"]))
                            continue
                        df = self._aggregate_hourly_to_daily(dfh)
                        for col in [
                            "volumeUSD",
                            "tvlUSD",
                            "token0Price",
                            "token1Price",
                            "sqrtPrice",
                        ]:
                            if col not in df:
                                df[col] = np.nan
                            df[col] = to_num(df[col])
                    else:
                        for col in [
                            "volumeUSD",
                            "tvlUSD",
                            "token0Price",
                            "token1Price",
                            "sqrtPrice",
                        ]:
                            if col not in df:
                                df[col] = np.nan
                            df[col] = to_num(df[col])

                # common normalization
                if "date" not in df:
                    raise RuntimeError("expected 'date' column after normalization")
                df["date"] = pd.to_datetime(df["date"], unit="s", utc=True).dt.floor(
                    "D"
                )
                df["pool_id"] = p["id"]
                df["chain"], df["version"] = p["chain"], p["version"]
                df["token0_symbol"] = (p.get("token0") or {}).get("symbol", "?")
                df["token1_symbol"] = (p.get("token1") or {}).get("symbol", "?")

                self.cache.save(ck, df)
                frames.append(df)

            except Exception as exc:
                logger.warning(f"daily fetch failed {p['id']}: {exc}")

        if skipped_missing_ts:
            eg = ", ".join([f"{c}:{pid}" for c, pid in skipped_missing_ts[:5]])
            logger.info(
                f"Skipped {len(skipped_missing_ts)} pools with no day/hour data since {since} "
                f"(e.g., {eg} …)"
            )
        if not frames:
            raise RuntimeError("no daily data fetched")
        return pd.concat(frames, ignore_index=True)

    # ------------------------------ IL calc -------------------------------- #
    @staticmethod
    def _il_pct(price_ratio: pd.Series | np.ndarray | float) -> pd.Series | float:
        """
        Impermanent-loss percentage for a constant-product LP.

        Parameters
        ----------
        price_ratio : float | pd.Series | np.ndarray
            p₁ / p₀ (new price divided by old price)

        Returns
        -------
        Same type as `price_ratio` containing IL%, with NaN where the
        ratio is non-positive or non-finite.
        """
        pr = np.asarray(price_ratio, dtype=float)
        il = (2 * np.sqrt(pr) / (1 + pr)) - 1
        il[(pr <= 0) | (~np.isfinite(pr))] = np.nan
        if isinstance(price_ratio, pd.Series):
            return pd.Series(il, index=price_ratio.index)
        return il

    def build_metrics(self, daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[pd.DataFrame] = []
        skipped_stable = 0
        for pid, df in daily.groupby("pool_id"):
            df = df.sort_values("date")

            # Choose a price series in order of preference, requiring >=2 valid points.
            price: Optional[pd.Series] = None
            if "token0Price" in df and df["token0Price"].notna().sum() >= 2:
                price = df.set_index("date")["token0Price"].astype(float)
            elif "token1Price" in df and df["token1Price"].notna().sum() >= 2:
                price = 1.0 / df.set_index("date")["token1Price"].astype(float)
            elif "sqrtPrice" in df and df["sqrtPrice"].notna().sum() >= 2:
                q96 = df.set_index("date")["sqrtPrice"].astype(float)
                price = (q96 / 2**96) ** 2

            if price is None:
                continue

            price = price.replace([np.inf, -np.inf], np.nan).dropna()
            if price.size < 2:
                continue

            # Exclude stable↔stable pools by median price ~ 1
            med_price = float(np.nanmedian(price.values))
            if (
                np.isfinite(med_price)
                and abs(med_price - 1.0) <= self.cfg.STABLE_PRICE_EPS
            ):
                skipped_stable += 1
                continue

            required_cols = {"volumeUSD", "tvlUSD"}
            if not required_cols.issubset(df.columns):
                continue

            vol = df.set_index("date").reindex(price.index)["volumeUSD"].astype(float)
            tvl = df.set_index("date").reindex(price.index)["tvlUSD"].astype(float)

            if vol.notna().sum() == 0 or tvl.notna().sum() == 0:
                continue

            il_pct = self._il_pct(price / price.shift(1))
            il_usd = il_pct * tvl

            metric_df = pd.DataFrame(
                {
                    "date": price.index,
                    "pool_id": pid,
                    "volume_usd": vol,
                    "il_usd": il_usd,
                    "chain": df["chain"].iloc[0],
                    "version": df["version"].iloc[0],
                    "pair": df["token0_symbol"].iloc[0]
                    + "/"
                    + df["token1_symbol"].iloc[0],
                    "median_price": med_price,
                }
            ).dropna(subset=["volume_usd", "il_usd"])

            if not metric_df.empty:
                rows.append(metric_df)

        if skipped_stable:
            logger.info(
                f"Excluded {skipped_stable} pools as stable↔stable (median price ≈ 1)"
            )

        if not rows:
            raise RuntimeError("no pools with valid IL + volume data")
        daily_metrics = pd.concat(rows, ignore_index=True)

        agg = daily_metrics.groupby(
            ["pool_id", "pair", "chain", "version"], as_index=False
        ).agg(
            avg_daily_volume_usd=("volume_usd", "mean"),
            avg_daily_il_usd=("il_usd", lambda s: float(np.mean(np.abs(s)))),
            median_price=("median_price", "median"),
            n_days=("il_usd", "count"),
        )

        agg = agg[agg["avg_daily_il_usd"] > self.cfg.MIN_AVG_ABS_IL_USD].copy()

        if self.cfg.MIN_AVG_DAILY_VOLUME_USD is not None:
            agg = agg[
                agg["avg_daily_volume_usd"] >= float(self.cfg.MIN_AVG_DAILY_VOLUME_USD)
            ]
        if self.cfg.MAX_AVG_DAILY_VOLUME_USD is not None:
            agg = agg[
                agg["avg_daily_volume_usd"] <= float(self.cfg.MAX_AVG_DAILY_VOLUME_USD)
            ]

        agg["il_per_1k_volume"] = (
            agg["avg_daily_il_usd"] / agg["avg_daily_volume_usd"]
        ) * 1000.0
        agg = agg[np.isfinite(agg["il_per_1k_volume"]) & (agg["il_per_1k_volume"] > 0)]

        survivors = set(agg["pool_id"].unique())
        daily_metrics = daily_metrics[daily_metrics["pool_id"].isin(survivors)].copy()

        return daily_metrics, agg

    # ------------------------------ outliers ------------------------------- #
    def filter_outliers(
        self, per_pool: pd.DataFrame, ratio_col: str = "il_per_1k_volume"
    ) -> pd.DataFrame:
        if per_pool.empty or ratio_col not in per_pool:
            return per_pool
        s_raw = per_pool[ratio_col].astype(float)
        if (s_raw < 0).any():
            logger.warning(f"{ratio_col} contains negative values; check pipeline.")
        lo, hi = mad_bounds_log(s_raw, k=self.cfg.MAD_K)
        lo = max(0.0, lo)
        before = per_pool.shape[0]
        filtered = per_pool[(s_raw >= lo) & (s_raw <= hi)].copy()
        removed = before - filtered.shape[0]
        logger.info(
            f"Outlier removal ({ratio_col}, log-MAD k={self.cfg.MAD_K}): "
            f"kept [{lo:.4g}, {hi:.4g}] → removed {removed}/{before}"
        )
        return filtered

    # ------------------------------ plots ---------------------------------- #
    def _plots(self, agg: pd.DataFrame) -> None:
        if agg.empty:
            return
        agg_sorted = agg.sort_values("il_per_1k_volume", ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            x=np.arange(len(agg_sorted)),
            height=agg_sorted["il_per_1k_volume"],
        )
        ax.set_xticks(np.arange(len(agg_sorted)))
        ax.set_xticklabels(agg_sorted["pair"], rotation=90, fontsize=6)
        ax.set_ylabel("USD |IL| per $1,000 of daily volume")
        ax.set_title("IL per $1k daily volume (per pool, averaged over lookback)")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(self.output / "bar_il_per_1k_volume.png", dpi=300)
        plt.close(fig)

    # ------------------------------ run ------------------------------------ #
    def run(self) -> None:
        pools = self.discover_pools()
        if not pools:
            raise RuntimeError(
                "discovery returned no pools; try different chains or discovery mode."
            )
        daily = self.fetch_daily(pools)
        daily_metrics, per_pool = self.build_metrics(daily)

        per_pool_filtered = self.filter_outliers(per_pool, ratio_col="il_per_1k_volume")

        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_pools_total": int(per_pool.shape[0]),
            "n_pools_after_filter": int(per_pool_filtered.shape[0]),
            "mean_il_per_1k_volume": (
                float(per_pool_filtered["il_per_1k_volume"].mean())
                if not per_pool_filtered.empty
                else float("nan")
            ),
            "median_il_per_1k_volume": (
                float(per_pool_filtered["il_per_1k_volume"].median())
                if not per_pool_filtered.empty
                else float("nan")
            ),
            "p90_il_per_1k_volume": (
                float(np.nanpercentile(per_pool_filtered["il_per_1k_volume"], 90))
                if not per_pool_filtered.empty
                else float("nan")
            ),
        }
        logger.info(json.dumps(summary, indent=2))

        self._plots(per_pool_filtered)

        daily_metrics.to_parquet(self.output / "daily_metrics.parquet", index=False)
        per_pool.to_csv(self.output / "per_pool_metrics_raw.csv", index=False)
        per_pool_filtered.to_csv(self.output / "per_pool_metrics.csv", index=False)
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.success(f"Results saved to {self.output}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Volume vs IL analysis for Uniswap pools")
    p.add_argument("--n-top", type=int, default=100)
    p.add_argument("--lookback-days", type=int, default=60)
    p.add_argument("--chains", nargs="+", default=["mainnet", "base"])
    p.add_argument("--output-dir", default="output")
    p.add_argument(
        "--stable-price-eps",
        type=float,
        default=0.03,
        help="Exclude pools with median price within [1±eps]",
    )
    p.add_argument(
        "--min-avg-abs-il-usd",
        type=float,
        default=1e-6,
        help="Drop pools whose avg |IL| is below this USD",
    )
    p.add_argument(
        "--mad-k",
        type=float,
        default=2.5,
        help="Log-MAD fence multiplier for outlier removal on il_per_1k_volume",
    )
    p.add_argument(
        "--discovery-mode",
        choices=["timeseries", "entity"],
        default="timeseries",
        help="Use day/hour entities to discover pools with in-window rows ('timeseries'), or lifetime volume ranking ('entity').",
    )
    p.add_argument(
        "--min-discovery-lifetime-volume-usd",
        type=float,
        default=None,
        help="Server-side prefilter on lifetime volumeUSD during entity-first discovery (optional).",
    )
    p.add_argument(
        "--max-discovery-lifetime-volume-usd",
        type=float,
        default=None,
        help="Server-side upper bound prefilter on lifetime volumeUSD during entity-first discovery (optional).",
    )
    p.add_argument(
        "--min-avg-daily-volume-usd",
        type=float,
        default=None,
        help="Exact filter on computed avg daily volume over lookback (no estimation).",
    )
    p.add_argument(
        "--max-avg-daily-volume-usd",
        type=float,
        default=None,
        help="Exact upper filter on computed avg daily volume over lookback (no estimation).",
    )

    args = p.parse_args()

    cfg = Config(
        n_top=args.n_top,
        lookback_days=args.lookback_days,
        chains=args.chains,
        output_dir=args.output_dir,
        stable_price_eps=args.stable_price_eps,
        min_avg_abs_il_usd=args.min_avg_abs_il_usd,
        mad_k=args.mad_k,
        discovery_mode=args.discovery_mode,
        min_discovery_lifetime_volume_usd=args.min_discovery_lifetime_volume_usd,
        max_discovery_lifetime_volume_usd=args.max_discovery_lifetime_volume_usd,
        min_avg_daily_volume_usd=args.min_avg_daily_volume_usd,
        max_avg_daily_volume_usd=args.max_avg_daily_volume_usd,
    )
    UniswapILVolumeAnalyser(cfg).run()


if __name__ == "__main__":
    main()
