#!/usr/bin/env python3
"""
SENTINEL System Launcher

Properly launches the SENTINEL system with correct Python path setup.

Usage:
    python3 run_sentinel.py [--config CONFIG] [--log-level LEVEL]
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Now import and run main
from src.main import main

if __name__ == "__main__":
    main()
