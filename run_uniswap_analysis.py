#!/usr/bin/env python3
"""
Wrapper script to run Uniswap V3 IL/Volume analysis from root directory.

Usage:
    python run_uniswap_analysis.py --days 30 --volume-min 0 --volume-max 10000 \
        --endpoint https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV \
        --valuation usd_current
"""

import sys
import subprocess
from pathlib import Path

def main():
    script_path = Path(__file__).parent / "uniswap_analysis" / "uniswap_v3_il_volume_analysis.py"
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()