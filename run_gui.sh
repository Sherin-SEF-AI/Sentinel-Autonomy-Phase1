#!/bin/bash
# SENTINEL GUI Launcher

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Set PYTHONPATH to src directory
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

echo "✓ PYTHONPATH set"
echo "Starting SENTINEL GUI..."

# Run GUI from project root (NOT from src directory)
python3 src/gui_main.py
