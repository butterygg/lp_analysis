#!/usr/bin/env python3
"""
Wrapper script to run chain comparison analysis from root directory.

Usage:
    python run_compare_chains.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    script_path = Path(__file__).parent / "comparison_tools" / "compare_all_chains.py"
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()