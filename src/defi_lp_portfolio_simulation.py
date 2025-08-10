#!/usr/bin/env python3
"""
High-level CLI entry-point (refactored).

It orchestrates:
• data collection
• periodic simulations
• stats + json dump
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from lp_simulation_utils import (
    PortfolioAnalyzer,
    SimulationConfig,
    cached_api_fetch,
    find_latest_timestamp,
    forward_fill_tvl_data,
    logging,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Convenience wrapper around chain TVL history
# ------------------------------------------------------------------ #
class ChainTVLRepository:
    """Download, cache, forward-fill & slice per-chain TVL series."""

    _LLAMA_ENDPOINT = "https://api.llama.fi/v2/historicalChainTvl/{chain}"

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        self.series_by_chain: Dict[str, Dict[int, float]] = {}

    # -------------------- public API -------------------- #
    def load_all(self) -> None:
        logger.info("Fetching TVL history for: %s", ", ".join(self.cfg.top_evm_chains))
        for chain in self.cfg.top_evm_chains:
            tvl = self._download_chain_history(chain)
            if tvl:
                self.series_by_chain[chain] = forward_fill_tvl_data(tvl)
        logger.info("✓ Loaded %d chains", len(self.series_by_chain))

    def slice_since(self, start_date: datetime) -> None:
        unix_start = int(start_date.timestamp())
        self.series_by_chain = {
            chain: {ts: val for ts, val in series.items() if ts >= unix_start}
            for chain, series in self.series_by_chain.items()
        }

    def chains(self) -> List[str]:
        return list(self.series_by_chain)

    def tvl(self, chain: str) -> Dict[int, float]:
        return self.series_by_chain.get(chain, {})

    # -------------------- internals -------------------- #
    def _download_chain_history(self, chain: str) -> Dict[int, float]:
        url = self._LLAMA_ENDPOINT.format(chain=chain)
        cache_path = Path("cache") / f"chain_{chain}_history.json"
        data = cached_api_fetch(url, cache_path)
        return {int(d["date"]): float(d["tvl"]) for d in data} if data else {}


# ------------------------------------------------------------------ #
# Per-chain portfolio simulator
# ------------------------------------------------------------------ #
class MultiChainPortfolio(PortfolioAnalyzer):
    """
    PortfolioAnalyzer subclass that loops over *multiple* chains and
    averages their contribution per timestamp.
    """

    def simulate_one_month_span(
        self,
        repo: ChainTVLRepository,
        span_start_ts: int,
        span_end_ts: int,
        debug: bool = False,
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict]:
        """
        Aggregate every chain’s returns for the span,
        normalising by number of participating chains at each timestamp.
        """
        aggregate: Dict[int, float] = {}
        aggregate_fee: Dict[int, float] = {}
        aggregate_il: Dict[int, float] = {}
        contributors: Dict[int, int] = {}
        dbg: Dict[str, Dict] = {}

        for chain in repo.chains():
            tvl_series = repo.tvl(chain)
            if not tvl_series:
                continue

            tot, fee, il, chain_dbg = super().simulate_single_period(
                tvl_series,
                span_start_ts,
                span_end_ts,
                chain_key=chain,
                debug=debug,
            )
            if not tot:
                continue

            for ts in tot:
                aggregate[ts] = aggregate.get(ts, 0) + tot[ts]
                aggregate_fee[ts] = aggregate_fee.get(ts, 0) + fee[ts]
                aggregate_il[ts] = aggregate_il.get(ts, 0) + il[ts]
                contributors[ts] = contributors.get(ts, 0) + 1
            dbg[chain] = chain_dbg

        # normalise by # chains
        for ts, count in contributors.items():
            aggregate[ts] /= count
            aggregate_fee[ts] /= count
            aggregate_il[ts] /= count

        return aggregate, aggregate_fee, aggregate_il, dbg


# ------------------------------------------------------------------ #
# Workflow façade – what gets called from `main()`
# ------------------------------------------------------------------ #
class SimulationWorkflow:
    def __init__(self) -> None:
        self.cfg = SimulationConfig()
        # inject custom price curves per chain
        self.cfg.chain_tvl_ratios = {
            "Arbitrum": dict(
                min_tvl_ratio=0.82,
                max_tvl_ratio=1.30,
                min_up_price=0.01,
                max_up_price=1.0,
            ),
            "Base": dict(
                min_tvl_ratio=0.82,
                max_tvl_ratio=1.40,
                min_up_price=0.01,
                max_up_price=1.0,
            ),
            "Unichain": dict(
                min_tvl_ratio=0.75,
                max_tvl_ratio=1.66,
                min_up_price=0.01,
                max_up_price=1.0,
            ),
        }
        self.cfg.top_evm_chains = ["Arbitrum", "Base", "Unichain"]

        self.repo = ChainTVLRepository(self.cfg)
        self.analyser = MultiChainPortfolio(self.cfg)

    def run(self) -> Dict:
        logger.info("=== LP PORTFOLIO SIMULATION ===")

        # -------- data ingestion -------- #
        self.repo.load_all()
        start_date = datetime.utcnow() - timedelta(days=self.cfg.analysis_months * 30)
        self.repo.slice_since(start_date)

        # -------- rolling 1-month windows -------- #
        perf, fee_perf, il_perf, dbg = self._run_overlapping_windows(start_date)

        # -------- stats -------- #
        totals = self.analyser.extract_final_returns(perf)
        fees = self.analyser.extract_final_returns(fee_perf)
        ils = self.analyser.extract_final_returns(il_perf)
        stats = self.analyser.summarise(totals, fees, ils)

        self._pretty_print(stats)
        return self._dump_json(stats, totals, fees, ils, dbg)

    # -------------------- internals -------------------- #
    def _run_overlapping_windows(
        self, first_window_start: datetime
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        logger.info("Simulating rolling windows …")
        perf_total, perf_fee, perf_il, dbg_all = {}, {}, {}, {}
        current = first_window_start
        last_start = first_window_start + timedelta(
            days=self.cfg.analysis_months * 30 - self.cfg.simulation_period_days
        )
        idx = 0
        while current <= last_start:
            span_end = current + timedelta(days=self.cfg.simulation_period_days)
            key = f"span_{idx:03d}_{current.date()}"

            if idx % 30 == 0:
                logger.info("→ %s", key)

            tot, fee, il, dbg = self.analyser.simulate_one_month_span(
                self.repo,
                int(current.timestamp()),
                int(span_end.timestamp()),
                debug=idx < 3,
            )
            perf_total[key], perf_fee[key], perf_il[key], dbg_all[key] = (
                tot,
                fee,
                il,
                dbg,
            )
            current += timedelta(days=self.cfg.period_spacing_days)
            idx += 1
        logger.info("✓ Generated %d windows", idx)
        return perf_total, perf_fee, perf_il, dbg_all

    @staticmethod
    def _pretty_print(stats: Dict) -> None:
        logger.info("\n=== SUMMARY ===")
        for section, vals in stats.items():
            logger.info("%s: %s", section.upper(), json.dumps(vals, indent=2))

    def _dump_json(
        self,
        stats: Dict,
        total: List[float],
        fee: List[float],
        il: List[float],
        dbg: Dict,
    ) -> Dict:
        out_dir = Path("portfolio_results")
        out_dir.mkdir(exist_ok=True)
        result = {
            "config": self.cfg.__dict__,
            "stats": stats,
            "final_total_returns": total,
            "final_fee_returns": fee,
            "final_il_returns": il,
            "debug": dbg,
        }
        (out_dir / "simulation_results.json").write_text(json.dumps(result, indent=2))
        logger.info("Results written to %s", out_dir / "simulation_results.json")
        return result


# ------------------------------------------------------------------ #
# CLI entry-point
# ------------------------------------------------------------------ #
def main() -> None:
    SimulationWorkflow().run()


if __name__ == "__main__":
    main()
