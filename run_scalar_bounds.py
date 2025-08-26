#!/usr/bin/env python3
"""
Wrapper script to run Scalar Bounds Calculator from root directory.

Usage:
    python run_scalar_bounds.py [chain] [period_days] [lookback_months]
    
Examples:
    python run_scalar_bounds.py Base 30 12
    python run_scalar_bounds.py Unichain 21 3
"""

import sys
import subprocess
from pathlib import Path

def main():
    script_path = Path(__file__).parent / "scalar_bounds_analysis" / "scalar_bounds_calculator.py"
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()