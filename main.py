#!/usr/bin/env python3
"""
ReID System — entry point.

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Ensure project root is on path regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from ui.app import main

if __name__ == "__main__":
    main()
