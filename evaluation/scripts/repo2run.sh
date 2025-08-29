#!/bin/bash

set -e

# Create output directory
mkdir -p build_output
chmod -R 777 .

# initial contents
printf '{"pytest": {}}\n' > build_output/results.json  

# If a bootstrap script exists, run it first
if [ -f "./bootstrap_script.sh" ]; then
  echo "Bootstrap script contents:"
  cat ./bootstrap_script.sh
  echo "Running bootstrap script..."
  source ./bootstrap_script.sh
fi

# Check that jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq not found"
    exit 1
fi

# install pytest
python -m pip install --quiet pytest pytest-json-report

# Print which Python is being used
echo "Using $(python --version) located at $(which python)"

# Run pytest
echo "Running pytest..."
python -m pytest --collect-only /data/project --json-report --json-report-file=build_output/pytest_output.json || true

# Check if pytest output exists and is valid JSON
if [ ! -f build_output/pytest_output.json ]; then
    echo "Failed to get valid pytest output"
    exit 1
fi

# Add pytest field to results
jq -s '.[0] * {"pytest": .[1]}' build_output/results.json build_output/pytest_output.json > \
    build_output/temp.json && mv build_output/temp.json build_output/results.json

chmod -R 777 .
exit 0