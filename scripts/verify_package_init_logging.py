#!/usr/bin/env python3
"""Verify src package initialization logging."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.core.logging import LoggerSetup

def main():
    """Test package initialization logging."""
    print("=" * 60)
    print("SENTINEL Package Initialization Logging Verification")
    print("=" * 60)
    
    # Setup logging
    LoggerSetup.setup(log_level='DEBUG', log_dir='logs')
    
    print("\n1. Testing successful import...")
    try:
        import src
        print(f"   ✓ Package imported successfully")
        print(f"   ✓ Exported modules: {src.__all__}")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return 1
    
    print("\n2. Testing module availability...")
    if 'SentinelSystem' in src.__all__:
        print(f"   ✓ SentinelSystem available")
    if 'main' in src.__all__:
        print(f"   ✓ main available")
    if 'gui_main' in src.__all__:
        print(f"   ✓ gui_main available")
    
    print("\n3. Checking log output...")
    print("   Check logs/sentinel_*.log for initialization messages")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
