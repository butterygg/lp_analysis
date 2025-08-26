#!/usr/bin/env python3
"""
Wrapper script to run LP visualization from root directory.

Usage:
    python run_visualization.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    script_path = Path(__file__).parent / "visualization" / "lp_visualization.py"
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()