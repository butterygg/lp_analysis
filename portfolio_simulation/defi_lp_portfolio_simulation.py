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
logger.level("DEBUG")


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
        cache_path = Path("../cache") / f"chain_{chain}_history.json"
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
    ) -> Tuple[
        Dict[int, float],
        Dict[int, float],
        Dict[int, float],
        Dict[int, float],
        Dict,
        Dict[str, Dict[int, float]],
        Dict[str, Dict[int, float]],
        Dict[str, Dict[int, float]],
        Dict[str, Dict[int, float]],
    ]:
        """
        Aggregate every chain's returns for the span,
        normalising by number of participating chains at each timestamp.
        Also returns individual chain results.
        """
        aggregate: Dict[int, float] = {}
        aggregate_fee: Dict[int, float] = {}
        aggregate_il: Dict[int, float] = {}
        aggregate_external: Dict[int, float] = {}
        contributors: Dict[int, int] = {}
        dbg: Dict[str, Dict] = {}

        # Store individual chain results
        chain_totals: Dict[str, Dict[int, float]] = {}
        chain_fees: Dict[str, Dict[int, float]] = {}
        chain_ils: Dict[str, Dict[int, float]] = {}
        chain_externals: Dict[str, Dict[int, float]] = {}

        for chain in repo.chains():
            tvl_series = repo.tvl(chain)
            if not tvl_series:
                continue

            tot, fee, il, external, chain_dbg = super().simulate_single_period(
                tvl_series,
                span_start_ts,
                span_end_ts,
                chain_key=chain,
                debug=debug,
            )
            if not tot:
                continue

            # Store individual chain results
            chain_totals[chain] = tot
            chain_fees[chain] = fee
            chain_ils[chain] = il
            chain_externals[chain] = external

            for ts in tot:
                aggregate[ts] = aggregate.get(ts, 0) + tot[ts]
                aggregate_fee[ts] = aggregate_fee.get(ts, 0) + fee[ts]
                aggregate_il[ts] = aggregate_il.get(ts, 0) + il[ts]
                aggregate_external[ts] = aggregate_external.get(ts, 0) + external[ts]
                contributors[ts] = contributors.get(ts, 0) + 1
            dbg[chain] = chain_dbg

        # normalise by # chains
        for ts, count in contributors.items():
            aggregate[ts] /= count
            aggregate_fee[ts] /= count
            aggregate_il[ts] /= count
            aggregate_external[ts] /= count

        return (
            aggregate,
            aggregate_fee,
            aggregate_il,
            aggregate_external,
            dbg,
            chain_totals,
            chain_fees,
            chain_ils,
            chain_externals,
        )


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
            "Hyperliquid L1": dict(
                min_tvl_ratio=0.75,
                max_tvl_ratio=1.73,
                min_up_price=0.01,
                max_up_price=1.0,
            ),
        }
        self.cfg.top_evm_chains = ["Arbitrum", "Base", "Unichain", "Hyperliquid L1"]

        self.repo = ChainTVLRepository(self.cfg)
        self.analyser = MultiChainPortfolio(self.cfg)

    def run(self) -> Dict:
        logger.info("=== LP PORTFOLIO SIMULATION ===")

        # -------- data ingestion -------- #
        self.repo.load_all()
        start_date = datetime.utcnow() - timedelta(days=self.cfg.analysis_months * 30)
        self.repo.slice_since(start_date)

        # -------- rolling 1-month windows -------- #
        (
            perf,
            fee_perf,
            il_perf,
            external_perf,
            dbg,
            chain_totals,
            chain_fees,
            chain_ils,
            chain_externals,
        ) = self._run_overlapping_windows(start_date)

        # -------- stats -------- #
        totals = self.analyser.extract_final_returns(perf)
        fees = self.analyser.extract_final_returns(fee_perf)
        ils = self.analyser.extract_final_returns(il_perf)
        externals = self.analyser.extract_final_returns(external_perf)

        # Debug: print the structure of the performance data
        logger.info("Debug: perf keys: %s", list(perf.keys())[:5] if perf else "None")
        logger.info(
            "Debug: fee_perf keys: %s",
            list(fee_perf.keys())[:5] if fee_perf else "None",
        )
        logger.info(
            "Debug: il_perf keys: %s", list(il_perf.keys())[:5] if il_perf else "None"
        )
        logger.info(
            "Debug: external_perf keys: %s",
            list(external_perf.keys())[:5] if external_perf else "None",
        )
        logger.info("Debug: totals length: %d", len(totals))
        logger.info("Debug: fees length: %d", len(fees))
        logger.info("Debug: ils length: %d", len(ils))
        logger.info("Debug: externals length: %d", len(externals))

        stats = self.analyser.summarise(totals, fees, ils, externals)

        # Calculate individual chain statistics
        chain_stats = self._calculate_individual_chain_stats(
            chain_totals, chain_fees, chain_ils, chain_externals
        )

        self._pretty_print(stats)
        self._print_individual_chain_results(chain_stats)
        self._print_portfolio_vs_individual_comparison(stats, chain_stats)
        self._print_summary_table(stats, chain_stats)

        # Calculate raw returns for each chain
        chain_raw_returns = {}
        for chain in chain_totals:
            chain_total_returns = self.analyser.extract_final_returns(
                chain_totals[chain]
            )
            chain_fee_returns = self.analyser.extract_final_returns(chain_fees[chain])
            chain_il_returns = self.analyser.extract_final_returns(chain_ils[chain])
            chain_external_returns = self.analyser.extract_final_returns(
                chain_externals[chain]
            )

            chain_raw_returns[chain] = {
                "total": [float(x * 100) for x in chain_total_returns],
                "fee": [float(x * 100) for x in chain_fee_returns],
                "il": [float(x * 100) for x in chain_il_returns],
                "external": [float(x * 100) for x in chain_external_returns],
            }

        return self._dump_json(
            stats, totals, fees, ils, externals, dbg, chain_stats, chain_raw_returns
        )

    # -------------------- internals -------------------- #
    def _run_overlapping_windows(self, first_window_start: datetime) -> Tuple[
        Dict,
        Dict,
        Dict,
        Dict,
        Dict,
        Dict[str, Dict],
        Dict[str, Dict],
        Dict[str, Dict],
        Dict[str, Dict],
    ]:
        logger.info("Simulating rolling windows …")
        perf_total, perf_fee, perf_il, perf_external, dbg_all = {}, {}, {}, {}, {}
        # Store individual chain results across all windows
        chain_totals_all: Dict[str, Dict] = {}
        chain_fees_all: Dict[str, Dict] = {}
        chain_ils_all: Dict[str, Dict] = {}
        chain_externals_all: Dict[str, Dict] = {}

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

            (
                tot,
                fee,
                il,
                external,
                dbg,
                chain_totals,
                chain_fees,
                chain_ils,
                chain_externals,
            ) = self.analyser.simulate_one_month_span(
                self.repo,
                int(current.timestamp()),
                int(span_end.timestamp()),
                debug=idx < 3,
            )
            (
                perf_total[key],
                perf_fee[key],
                perf_il[key],
                perf_external[key],
                dbg_all[key],
            ) = (
                tot,
                fee,
                il,
                external,
                dbg,
            )

            # Store individual chain results for this window
            for chain in chain_totals:
                if chain not in chain_totals_all:
                    chain_totals_all[chain] = {}
                    chain_fees_all[chain] = {}
                    chain_ils_all[chain] = {}
                    chain_externals_all[chain] = {}
                chain_totals_all[chain][key] = chain_totals[chain]
                chain_fees_all[chain][key] = chain_fees[chain]
                chain_ils_all[chain][key] = chain_ils[chain]
                chain_externals_all[chain][key] = chain_externals[chain]

            current += timedelta(days=self.cfg.period_spacing_days)
            idx += 1
        logger.info("✓ Generated %d windows", idx)
        return (
            perf_total,
            perf_fee,
            perf_il,
            perf_external,
            dbg_all,
            chain_totals_all,
            chain_fees_all,
            chain_ils_all,
            chain_externals_all,
        )

    def _calculate_individual_chain_stats(
        self,
        chain_totals: Dict[str, Dict[str, Dict[int, float]]],
        chain_fees: Dict[str, Dict[str, Dict[int, float]]],
        chain_ils: Dict[str, Dict[str, Dict[int, float]]],
        chain_externals: Dict[str, Dict[str, Dict[int, float]]],
    ) -> Dict[str, Dict]:
        """
        Calculates individual chain statistics (e.g., total returns, fees, IL)
        for each window across all chains.
        """
        chain_stats: Dict[str, Dict] = {}
        for chain in chain_totals:
            chain_stats[chain] = {}
            for window_key in chain_totals[chain]:
                # Use end-of-window returns (consistent with portfolio summary), in percent
                def last_pct(series: Dict[int, float]) -> float:
                    if not series:
                        return 0.0
                    last_ts = max(series)
                    return float(series[last_ts] * 100.0)

                chain_stats[chain][window_key] = {
                    "total_returns": last_pct(chain_totals[chain][window_key]),
                    "fee_returns": last_pct(chain_fees[chain][window_key]),
                    "il_returns": last_pct(chain_ils[chain][window_key]),
                    "external_returns": last_pct(chain_externals[chain][window_key]),
                }
        return chain_stats

    def _print_individual_chain_results(self, chain_stats: Dict[str, Dict]) -> None:
        """
        Prints individual chain statistics for each window.
        """
        logger.info("\n=== INDIVIDUAL CHAIN RESULTS ===")
        for chain, windows in chain_stats.items():
            logger.info("\nChain: %s", chain)

            # Calculate summary statistics across all windows
            all_total_returns = [stats["total_returns"] for stats in windows.values()]
            all_fee_returns = [stats["fee_returns"] for stats in windows.values()]
            all_il_returns = [stats["il_returns"] for stats in windows.values()]
            all_external_returns = [
                stats["external_returns"] for stats in windows.values()
            ]

            logger.info("  Summary across all windows:")
            logger.info(
                "    Total Returns: avg=%.4f, min=%.4f, max=%.4f",
                np.mean(all_total_returns),
                np.min(all_total_returns),
                np.max(all_total_returns),
            )
            logger.info(
                "    Fee Returns:  avg=%.4f, min=%.4f, max=%.4f",
                np.mean(all_fee_returns),
                np.min(all_fee_returns),
                np.max(all_fee_returns),
            )
            logger.info(
                "    IL Returns:   avg=%.4f, min=%.4f, max=%.4f",
                np.mean(all_il_returns),
                np.min(all_il_returns),
                np.max(all_il_returns),
            )
            logger.info(
                "    External Returns: avg=%.4f, min=%.4f, max=%.4f",
                np.mean(all_external_returns),
                np.min(all_external_returns),
                np.max(all_external_returns),
            )

            # Show first few windows for detail
            logger.info("  First 3 windows detail:")
            for i, (window_key, stats) in enumerate(windows.items()):
                if i >= 3:
                    break
                logger.info(
                    "    %s: Total=%.4f, Fees=%.4f, IL=%.4f, External=%.4f",
                    window_key,
                    stats["total_returns"],
                    stats["fee_returns"],
                    stats["il_returns"],
                    stats["external_returns"],
                )

    def _print_portfolio_vs_individual_comparison(
        self, portfolio_stats: Dict, chain_stats: Dict[str, Dict]
    ) -> None:
        """
        Prints a comparison of the portfolio's performance with individual chain performance.
        """
        logger.info("\n=== PORTFOLIO VS INDIVIDUAL CHAIN COMPARISON ===")

        # Extract portfolio summary stats
        portfolio_total_avg = portfolio_stats.get("total", {}).get("mean_pct", 0)
        portfolio_fee_avg = portfolio_stats.get("fee", {}).get("mean_pct", 0)
        portfolio_il_avg = portfolio_stats.get("il", {}).get("mean_pct", 0)
        portfolio_external_avg = portfolio_stats.get("external", {}).get("mean_pct", 0)

        logger.info("Equal-Weighted Portfolio (across all chains):")
        logger.info("  Total Returns: %.4f", portfolio_total_avg)
        logger.info("  Fee Returns:  %.4f", portfolio_fee_avg)
        logger.info("  IL Returns:   %.4f", portfolio_il_avg)
        logger.info("  External Returns: %.4f", portfolio_external_avg)

        logger.info("\nIndividual Chain Performance vs Portfolio:")
        for chain, windows in chain_stats.items():
            # Calculate average returns across all windows for this chain
            all_total_returns = [stats["total_returns"] for stats in windows.values()]
            all_fee_returns = [stats["fee_returns"] for stats in windows.values()]
            all_il_returns = [stats["il_returns"] for stats in windows.values()]
            all_external_returns = [
                stats["external_returns"] for stats in windows.values()
            ]

            chain_total_avg = np.mean(all_total_returns)
            chain_fee_avg = np.mean(all_fee_returns)
            chain_il_avg = np.mean(all_il_returns)
            chain_external_avg = np.mean(all_external_returns)

            logger.info("\n%s:", chain)
            logger.info(
                "  Total Returns: %.4f (vs portfolio: %+.4f)",
                chain_total_avg,
                chain_total_avg - portfolio_total_avg,
            )
            logger.info(
                "  Fee Returns:  %.4f (vs portfolio: %+.4f)",
                chain_fee_avg,
                chain_fee_avg - portfolio_fee_avg,
            )
            logger.info(
                "  IL Returns:   %.4f (vs portfolio: %+.4f)",
                chain_il_avg,
                chain_il_avg - portfolio_il_avg,
            )
            logger.info(
                "  External Returns: %.4f (vs portfolio: %+.4f)",
                chain_external_avg,
                chain_external_avg - portfolio_external_avg,
            )

    def _print_summary_table(
        self, portfolio_stats: Dict, chain_stats: Dict[str, Dict]
    ) -> None:
        """
        Prints a summary table comparing the portfolio's performance with individual chain performance.
        """
        logger.info("\n=== SUMMARY TABLE ===")
        logger.info(
            "| Chain | Total Returns (Avg) | Fee Returns (Avg) | IL Returns (Avg) | External Returns (Avg) |"
        )
        logger.info(
            "|-------|--------------------|--------------------|--------------------|--------------------|"
        )

        portfolio_total_avg = portfolio_stats.get("total", {}).get("mean_pct", 0)
        portfolio_fee_avg = portfolio_stats.get("fee", {}).get("mean_pct", 0)
        portfolio_il_avg = portfolio_stats.get("il", {}).get("mean_pct", 0)
        portfolio_external_avg = portfolio_stats.get("external", {}).get("mean_pct", 0)

        for chain, windows in chain_stats.items():
            all_total_returns = [stats["total_returns"] for stats in windows.values()]
            all_fee_returns = [stats["fee_returns"] for stats in windows.values()]
            all_il_returns = [stats["il_returns"] for stats in windows.values()]
            all_external_returns = [
                stats["external_returns"] for stats in windows.values()
            ]

            chain_total_avg = np.mean(all_total_returns)
            chain_fee_avg = np.mean(all_fee_returns)
            chain_il_avg = np.mean(all_il_returns)
            chain_external_avg = np.mean(all_external_returns)

            logger.info(
                "| %s | %.4f (vs portfolio: %+.4f) | %.4f (vs portfolio: %+.4f) | %.4f (vs portfolio: %+.4f) | %.4f (vs portfolio: %+.4f) |"
                % (
                    chain,
                    chain_total_avg,
                    chain_total_avg - portfolio_total_avg,
                    chain_fee_avg,
                    chain_fee_avg - portfolio_fee_avg,
                    chain_il_avg,
                    chain_il_avg - portfolio_il_avg,
                    chain_external_avg,
                    chain_external_avg - portfolio_external_avg,
                )
            )

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
        external: List[float],
        dbg: Dict,
        chain_stats: Dict[str, Dict],
        chain_raw_returns: Dict[str, Dict[str, List[float]]],
    ) -> Dict:
        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(exist_ok=True)

        # Create config with exact structure as specified
        config = {
            "analysis_months": self.cfg.analysis_months,
            "simulation_period_days": self.cfg.simulation_period_days,
            "period_spacing_days": self.cfg.period_spacing_days,
            "fee_rate": self.cfg.fee_rate,
            "withdrawal_enabled": self.cfg.withdrawal_enabled,
            "withdrawal_timing_pct": self.cfg.withdrawal_timing_pct,
            "withdrawal_amount_pct": self.cfg.withdrawal_amount_pct,
            "chain_tvl_ratios": self.cfg.chain_tvl_ratios,
            "top_evm_chains": self.cfg.top_evm_chains,
        }

        result = {
            "config": config,
            "stats": stats,
            "raw_returns": {
                "total": [float(x * 100) for x in total],  # Convert to percentage
                "fee": [float(x * 100) for x in fee],  # Convert to percentage
                "il": [float(x * 100) for x in il],  # Convert to percentage
                "external": [float(x * 100) for x in external],  # Convert to percentage
            },
            "chain_stats": chain_stats,
            "per_chain_raw_returns": chain_raw_returns,
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
