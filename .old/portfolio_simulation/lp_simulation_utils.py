#!/usr/bin/env python3
"""
Shared utilities for LP Portfolio simulations (refactored).

The module keeps every original feature but is rewritten for:
• readability
• lower cyclomatic complexity
• single-responsibility helpers
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


# ------------------------------------------------------------------ #
# Configuration dataclass
# ------------------------------------------------------------------ #
@dataclass
class SimulationConfig:
    """Runtime parameters used by every helper in the project."""

    analysis_months: int = 12
    simulation_period_days: int = 21
    period_spacing_days: int = 1
    fee_rate: float = 0.003
    withdrawal_enabled: bool = False
    withdrawal_timing_pct: float = 0.25
    withdrawal_amount_pct: float = 0.7
    chain_tvl_ratios: Dict[str, Dict[str, float]] | None = None  # injected later

    def __post_init__(self) -> None:
        self.chain_tvl_ratios = self.chain_tvl_ratios or {}


# ------------------------------------------------------------------ #
# I/O helpers
# ------------------------------------------------------------------ #
def cached_api_fetch(url: str, cache_path: Path) -> Dict:
    """
    GET `url`, cache to `cache_path`, and always return a decoded json object.

    Caching avoids hammering the DeFiLlama API when iterating on notebooks.
    """
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists():
        logger.debug("Cache hit for %s", url)
        return json.loads(cache_path.read_text())

    logger.info("Cache miss – pulling %s", url)
    response = requests.get(url, timeout=30)
    if not response.ok:
        logger.error("Failed to fetch %s: HTTP %s", url, response.status_code)
        return {}

    cache_path.write_text(response.text)
    return response.json()


# ------------------------------------------------------------------ #
# Core mathematics for AMM & portfolio
# ------------------------------------------------------------------ #
class LPPoolSimulator:
    """Everything needed to mark-to-market an UP/DOWN liquidity pool."""

    _INITIAL_TOKEN_MINT: float = 1_000.0  # 1k UP & 1k DOWN are minted for every run

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg

    # -------------------- price mechanics -------------------- #
    def get_token_prices(
        self, tvl_ratio: float, chain_key: str = "default"
    ) -> Tuple[float, float]:
        """
        Piece-wise-linear mapping: TVL ratio  →  UP price.

        DOWN price is simply 1-UP (they always sum to $1 here).
        """
        params = self._get_chain_price_params(chain_key)
        up_price = np.interp(
            tvl_ratio,
            (params["min_tvl_ratio"], params["max_tvl_ratio"]),
            (params["min_up_price"], params["max_up_price"]),
        )
        up_price = np.clip(up_price, params["min_up_price"], params["max_up_price"])

        # Ensure prices are always positive and sum to 1
        up_price = max(up_price, 0.001)  # Minimum 0.1% price
        up_price = min(up_price, 0.999)  # Maximum 99.9% price
        down_price = 1.0 - up_price

        return up_price, down_price

    # -------------------- pool bootstrap -------------------- #
    def bootstrap_balances(
        self, chain_key: str = "default"
    ) -> Tuple[float, float, float, float]:
        """
        Mint 1k/1k tokens and deposit equal USD value into the AMM.

        Returns:
            up_in_pool, down_in_pool, up_external, down_external
        """
        price_up, price_down = self.get_token_prices(1.0, chain_key)

        # Safety check: ensure prices are valid
        if price_up <= 0 or price_down <= 0:
            # Fallback to equal distribution
            price_up, price_down = 0.5, 0.5

        # Maximum USD value from each side
        max_usd_up = self._INITIAL_TOKEN_MINT * price_up
        max_usd_down = self._INITIAL_TOKEN_MINT * price_down
        usd_per_side = min(max_usd_up, max_usd_down)

        up_in_pool = usd_per_side / price_up
        down_in_pool = usd_per_side / price_down
        up_external = self._INITIAL_TOKEN_MINT - up_in_pool
        down_external = self._INITIAL_TOKEN_MINT - down_in_pool
        return up_in_pool, down_in_pool, up_external, down_external

    # -------------------- withdraw & fee calculus -------------------- #
    def execute_withdrawal(
        self, invariant_k: float, price_up: float, price_down: float
    ) -> Tuple[float, float, float]:
        """Remove `withdrawal_amount_pct` liquidity once per period."""
        if min(price_up, price_down) <= 0:
            return 0.0, 0.0, invariant_k

        # Safety check: prevent division by zero
        if price_up <= 0 or price_down <= 0:
            return 0.0, 0.0, invariant_k

        pool_up = np.sqrt(invariant_k * price_down / price_up)
        pool_down = np.sqrt(invariant_k * price_up / price_down)

        # Check for invalid values
        if np.any(np.isnan([pool_up, pool_down])) or np.any(
            np.isinf([pool_up, pool_down])
        ):
            return 0.0, 0.0, invariant_k

        up_out = pool_up * self.cfg.withdrawal_amount_pct
        down_out = pool_down * self.cfg.withdrawal_amount_pct
        new_k = invariant_k * (1 - self.cfg.withdrawal_amount_pct) ** 2
        return up_out, down_out, new_k

    @staticmethod
    def constant_product_value(k: float, price_up: float, price_down: float) -> float:
        """USD value of a constant-product pool."""
        if min(price_up, price_down) <= 0:
            return 0.0

        # Safety check: prevent division by zero
        if price_up <= 0 or price_down <= 0:
            return 0.0

        up_amt = np.sqrt(k * price_down / price_up)
        down_amt = np.sqrt(k * price_up / price_down)

        # Check for invalid values
        if np.any(np.isnan([up_amt, down_amt])) or np.any(np.isinf([up_amt, down_amt])):
            return 0.0

        return up_amt * price_up + down_amt * price_down

    # -------------------- one-period simulation -------------------- #
    def simulate_one_period(
        self,
        tvl_by_timestamp: Dict[int, float],
        period_start: int,
        period_end: int,
        chain_key: str = "default",
    ) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]], float]:
        """
        Drive the pool across every timestamp inside [start, end].

        Returns:
            pool_value_by_ts, external_balances_by_ts, accumulated_fees
        """
        timeline = sorted(
            ts for ts in tvl_by_timestamp if period_start <= ts <= period_end
        )
        if len(timeline) < 2:
            return {}, {}, 0.0

        start_tvl = tvl_by_timestamp[timeline[0]]
        if start_tvl <= 0:
            return {}, {}, 0.0

        # ---- initial state --------------------------------------------------- #
        (
            pool_up,
            pool_down,
            ext_up,
            ext_down,
        ) = self.bootstrap_balances(chain_key)
        invariant_k = pool_up * pool_down

        # bookkeeping
        pool_value_by_ts: Dict[int, float] = {}
        external_by_ts: Dict[int, Tuple[float, float]] = {}
        fees_accumulated = 0.0
        last_ts, last_price_up, last_price_down = timeline[0], None, None

        # deterministic withdrawal target
        withdraw_at = period_start + int(
            (period_end - period_start) * self.cfg.withdrawal_timing_pct
        )
        withdrew = False

        # ---- walk through time ------------------------------------------------ #
        for ts in timeline:
            tvl_ratio = tvl_by_timestamp[ts] / start_tvl
            price_up, price_down = self.get_token_prices(tvl_ratio, chain_key)

            # fees from virtual volume between last snapshot and now
            fees_accumulated += self._fees_since_last_tick(
                invariant_k,
                price_up,
                price_down,
                ts,
                last_ts,
                last_price_up,
                last_price_down,
            )
            last_ts, last_price_up, last_price_down = ts, price_up, price_down

            # optional withdrawal once per period
            if self._should_withdraw(ts, withdrew, withdraw_at):
                up_out, down_out, invariant_k = self.execute_withdrawal(
                    invariant_k, price_up, price_down
                )
                ext_up += up_out
                ext_down += down_out
                withdrew = True

            pool_value_by_ts[ts] = self.constant_product_value(
                invariant_k, price_up, price_down
            )
            external_by_ts[ts] = (ext_up, ext_down)

        return pool_value_by_ts, external_by_ts, fees_accumulated

    # -------------------- internal pure helpers -------------------- #
    def _get_chain_price_params(self, key: str) -> Dict[str, float]:
        """Return chain-specific pricing params, defaulting gracefully."""
        if not self.cfg.chain_tvl_ratios:
            # Fallback to reasonable defaults if no chain params exist
            return {
                "min_tvl_ratio": 0.1,
                "max_tvl_ratio": 10.0,
                "min_up_price": 0.001,
                "max_up_price": 0.999,
            }

        if key not in self.cfg.chain_tvl_ratios:
            fallback = next(iter(self.cfg.chain_tvl_ratios))
            logger.warning("Unknown chain '%s' – falling back to '%s'", key, fallback)
            return self.cfg.chain_tvl_ratios[fallback]
        return self.cfg.chain_tvl_ratios[key]

    def _fees_since_last_tick(
        self,
        k: float,
        price_up: float,
        price_down: float,
        ts_now: int,
        ts_prev: int,
        prev_up: float | None,
        prev_down: float | None,
    ) -> float:
        """Implied fee revenue between two price points."""
        if prev_up is None or ts_now == ts_prev:
            return 0.0

        # Safety check: prevent division by zero or negative prices
        if price_up <= 0 or price_down <= 0 or prev_up <= 0 or prev_down <= 0:
            return 0.0

        # virtual token amounts now vs then
        up_now = np.sqrt(k * price_down / price_up)
        up_prev = np.sqrt(k * prev_down / prev_up)

        # Check for invalid values
        if np.any(np.isnan([up_now, up_prev])) or np.any(np.isinf([up_now, up_prev])):
            return 0.0

        volume_usd = (abs(up_now - up_prev) * price_up) / 2.0
        return volume_usd * self.cfg.fee_rate

    def _should_withdraw(self, ts: int, already_withdrew: bool, target_ts: int) -> bool:
        """Single point-in-time withdrawal guard."""
        return self.cfg.withdrawal_enabled and not already_withdrew and ts >= target_ts


# ------------------------------------------------------------------ #
# Portfolio analytics
# ------------------------------------------------------------------ #
def find_latest_timestamp(candidates: List[int], target: int) -> int | None:
    """Return max(ts) ≤ target or `None`."""
    before_target = [ts for ts in candidates if ts <= target]
    return max(before_target) if before_target else None


def forward_fill_tvl_data(
    raw_series: Dict[int, float], max_gap_days: int = 7
) -> Dict[int, float]:
    """
    Easy “DB forward fill” for gaps ≤ `max_gap_days` days.
    Keeps the original resolution of 1 day.
    """
    if len(raw_series) < 2:
        return raw_series

    filled = dict(raw_series)  # mutable copy
    seconds_per_day = 86_400
    sorted_ts = sorted(raw_series)

    for t_curr, t_next in zip(sorted_ts, sorted_ts[1:]):
        gap = (t_next - t_curr) // seconds_per_day
        if 1 < gap <= max_gap_days:
            price_curr = raw_series[t_curr]
            for i in range(1, gap):
                filled[t_curr + i * seconds_per_day] = price_curr
    return filled


# ------------------------------------------------------------------ #
# Higher-level portfolio wrapper
# ------------------------------------------------------------------ #
class PortfolioAnalyzer:
    """
    Builds on LPPoolSimulator to calculate total, fee, and IL returns.
    """

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        self.pool = LPPoolSimulator(cfg)

    # -------------------- public API -------------------- #
    def simulate_single_period(
        self,
        tvl_by_timestamp: Dict[int, float],
        start_ts: int,
        end_ts: int,
        chain_key: str = "default",
        debug: bool = False,
    ) -> Tuple[
        Dict[int, float],
        Dict[int, float],
        Dict[int, float],
        Dict[int, float],
        Dict[str, float],
    ]:
        pool_val, ext_balances, fees = self.pool.simulate_one_period(
            tvl_by_timestamp, start_ts, end_ts, chain_key
        )
        if not pool_val:
            return {}, {}, {}, {}, {}

        timeline = sorted(pool_val)
        returns_total, returns_fee, returns_il, returns_external = (
            self._build_return_series(
                pool_val, ext_balances, tvl_by_timestamp, timeline, fees, chain_key
            )
        )
        dbg = self._debug_metrics(tvl_by_timestamp, timeline, returns_il, chain_key)
        if debug:
            self._print_debug_snapshot(dbg)

        return returns_total, returns_fee, returns_il, returns_external, dbg

    # -------------------- stats helpers -------------------- #
    def _build_return_series(
        self,
        pool_val_by_ts: Dict[int, float],
        external_by_ts: Dict[int, Tuple[float, float]],
        tvl_by_ts: Dict[int, float],
        timeline: List[int],
        fees: float,
        chain_key: str,
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
        start_tvl = tvl_by_ts[timeline[0]]
        if start_tvl <= 0:
            return {}, {}, {}, {}

        # Common initial notional (USD)
        initial_notional = 1_000.0

        ret_total: Dict[int, float] = {}
        ret_fee: Dict[int, float] = {}
        ret_il: Dict[int, float] = {}
        ret_external: Dict[int, float] = {}

        # Prices at start and initial external USD value (captures 1) un-deposited tokens)
        start_price_up, start_price_down = self.pool.get_token_prices(1.0, chain_key)
        ext_up0, ext_down0 = external_by_ts[timeline[0]]
        initial_external_usd = ext_up0 * start_price_up + ext_down0 * start_price_down

        for ts in timeline:
            tvl_ratio = tvl_by_ts[ts] / start_tvl
            price_up, price_down = self.pool.get_token_prices(tvl_ratio, chain_key)

            ext_up, ext_down = external_by_ts[ts]
            ext_value = ext_up * price_up + ext_down * price_down
            pool_value = pool_val_by_ts[ts]

            # Fee component on initial notional (fees are cumulative for the period)
            ret_fee[ts] = fees / initial_notional

            # Total return on initial notional
            gross_value = pool_value + ext_value
            ret_total[ts] = (gross_value + fees - initial_notional) / initial_notional

            # External component: change in value of tokens held outside the AMM
            ret_external[ts] = (ext_value - initial_external_usd) / initial_notional

            # IL component as residual so that: total == fee + il + external
            ret_il[ts] = ret_total[ts] - ret_fee[ts] - ret_external[ts]

        return ret_total, ret_fee, ret_il, ret_external

    def _debug_metrics(
        self,
        tvl_by_ts: Dict[int, float],
        timeline: List[int],
        il_returns: Dict[int, float],
        chain_key: str,
    ) -> Dict[str, float]:
        start_tvl, end_tvl = tvl_by_ts[timeline[0]], tvl_by_ts[timeline[-1]]
        price_up_start, price_down_start = self.pool.get_token_prices(1.0, chain_key)
        price_up_end, price_down_end = self.pool.get_token_prices(
            end_tvl / start_tvl, chain_key
        )

        # Safety check: prevent division by zero
        if price_down_start <= 0 or price_down_end <= 0:
            ratio_start = 1.0
            ratio_end = 1.0
        else:
            ratio_start = price_up_start / price_down_start
            ratio_end = price_up_end / price_down_end

        return {
            "start_tvl": start_tvl,
            "end_tvl": end_tvl,
            "tvl_change_pct": (end_tvl - start_tvl) / start_tvl * 100,
            "up_down_ratio_change_pct": (ratio_end - ratio_start) / ratio_start * 100,
            "il_return_pct": il_returns[timeline[-1]] * 100,
        }

    @staticmethod
    def _print_debug_snapshot(dbg: Dict[str, float]) -> None:
        logger.info(
            "TVL Δ: %.2f%% | UP:DOWN Δ: %.2f%% | IL: %.2f%%",
            dbg["tvl_change_pct"],
            dbg["up_down_ratio_change_pct"],
            dbg["il_return_pct"],
        )

    # -------------------- aggregated utils -------------------- #
    @staticmethod
    def extract_final_returns(perfs: Dict[str, Dict[int, float]]) -> List[float]:
        return [
            series[max(series)]
            for series in perfs.values()
            if len(series) >= 2 and max(series) in series
        ]

    @staticmethod
    def _percentiles(arr: np.ndarray) -> Dict[str, float]:
        return {f"p{p}": float(np.percentile(arr, p) * 100) for p in (10, 25, 50, 75)}

    def summarise(
        self,
        final_total: List[float],
        final_fee: List[float] | None = None,
        final_il: List[float] | None = None,
        final_external: List[float] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        if not final_total:
            return {}

        # Filter out invalid values
        valid_total = [x for x in final_total if np.isfinite(x)]
        if not valid_total:
            return {}

        res: Dict[str, Dict[str, float]] = {}
        total_arr = np.array(valid_total)
        res["total"] = {
            "mean_pct": float(total_arr.mean() * 100),
            "std_pct": float(total_arr.std() * 100),
            **self._percentiles(total_arr),
        }

        def section(name: str, data: List[float] | None) -> None:
            if data:
                valid_data = [x for x in data if np.isfinite(x)]
                if valid_data:
                    arr = np.array(valid_data)
                    res[name] = {
                        "mean_pct": float(arr.mean() * 100),
                        **self._percentiles(arr),
                    }

        section("fee", final_fee)
        section("il", final_il)
        section("external", final_external)
        return res
