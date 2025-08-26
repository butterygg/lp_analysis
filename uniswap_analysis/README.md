# Uniswap V3 IL/Volume Analysis

This module analyzes impermanent loss (IL) vs volume for Uniswap V3 pools.

## Files
- `uniswap_v3_il_volume_analysis.py` - Main analysis script
- `pools_il_volume.csv` - Output: per-pool summary data
- `pools_il_volume_distribution.png` - Output: histogram of IL/Volume ratios
- `test_small.csv` - Small test dataset
- `test_small_distribution.png` - Test distribution visualization

## Usage

From the repository root:
```bash
python run_uniswap_analysis.py --days 30 --volume-min 0 --volume-max 10000 \
    --endpoint https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV \
    --valuation usd_current
```

Or directly:
```bash
cd uniswap_analysis
python uniswap_v3_il_volume_analysis.py [options]
```

## Outputs
- CSV with per-pool IL/Volume metrics
- PNG histogram showing distribution of ratios
- Console statistics and top pools by various metrics