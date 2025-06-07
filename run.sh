#!/bin/bash
# run.sh

# Activate virtual environment
source venv/bin/activate

# Run Python script with arguments
python3 solution.py "$1" "$2" "$3"

# Deactivate (optional but clean)
deactivate
