#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
VENV_DIR="$PROJECT_ROOT/_bin/venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Please run 'make setup' first."
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Run the backtester with all arguments passed to this script
python "$SCRIPT_DIR/backtest.py" "$@"

# Deactivate virtual environment
deactivate 