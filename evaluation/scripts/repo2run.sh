#!/bin/bash

set -e

# Create output directory
mkdir -p build_output
chmod -R 777 .

# If a bootstrap script exists, run it first
if [ -f "./bootstrap_script.sh" ]; then
  echo "Bootstrap script contents:"
  cat ./bootstrap_script.sh
  echo "Running bootstrap script..."
  source ./bootstrap_script.sh
fi

# install pytest
python -m pip install --quiet pytest

# Print which Python is being used
echo "Using $(python --version) located at $(which python)"

# Run pytest
echo "Running pytest..."
if ! command -v pytest &> /dev/null; then
    echo "pytest not found"
    exit 1
fi

# Run pyright and capture its output regardless of exit code
python -m pytest --collect-only -q /data/project > build_output/pytest_output.txt || true

# Check if pyright output exists and is valid JSON
if [ ! -f build_output/pytest_output.txt ]; then
    echo "Failed to get valid pytest output"
    exit 1
fi

chmod -R 777 .
exit 0 