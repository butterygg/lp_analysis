#!/usr/bin/env python3
"""
Test script to verify chain-specific TVL ratio configurations work correctly.
"""

from src.lp_simulation_utils import SimulationConfig, LPPoolSimulator


def test_chain_specific_configs():
    """Test that different chains use different TVL ratio configurations."""

    # Create config with chain-specific TVL ratios
    config = SimulationConfig()

    # Create simulator
    simulator = LPPoolSimulator(config)

    # Test different TVL ratios for different chains
    test_tvl_ratios = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print("Testing chain-specific TVL ratio configurations:")
    print("=" * 60)

    for chain in ["default", "Arbitrum", "Base", "Unichain"]:
        print(f"\nChain: {chain}")
        print("-" * 30)

        for tvl_ratio in test_tvl_ratios:
            up_price, down_price = simulator.calculate_token_prices(tvl_ratio, chain)
            print(
                f"  TVL ratio {tvl_ratio:3.1f}: UP={up_price:6.3f}, DOWN={down_price:6.3f}"
            )

    print("\n" + "=" * 60)
    print("Configuration used:")
    for chain, config_data in config.chain_tvl_ratios.items():
        print(
            f"{chain}: min_ratio={config_data['min_tvl_ratio']}, max_ratio={config_data['max_tvl_ratio']}"
        )


if __name__ == "__main__":
    test_chain_specific_configs()
