#!/bin/bash

set -e

# Create output directory
mkdir -p build_output
chmod -R 777 .
# initial contents
printf '{"pyright": {}}\n' > build_output/results.json  

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

# install pyright
python -m pip install --quiet pyright

# Print which Python is being used
echo "Using $(python --version) located at $(which python)"

# Run type checking with pyright
echo "Running type checks..."
if ! command -v pyright &> /dev/null; then
    echo "pyright not found"
    exit 1
fi

# Run pyright and capture its output regardless of exit code
python -m pyright /data/project --level error --outputjson > build_output/pyright_output.json || true

# Check if pyright output exists and is valid JSON
if [ ! -f build_output/pyright_output.json ]; then
    echo "Failed to get valid pyright output"
    exit 1
fi

# Count only critical import issues (reportMissingImports)
issue_count=$(jq '[.generalDiagnostics[] | select(.rule == "reportMissingImports")] | length' \
    build_output/pyright_output.json)

# Output results as JSON - exit code is 0 if pyright ran successfully, regardless of found issues
jq --arg issues "$issue_count" '. + {"issues_count": ($issues|tonumber)}' \
    build_output/results.json > build_output/temp.json && \
    mv build_output/temp.json build_output/results.json

# Add pyright field to results
jq -s '.[0] * {"pyright": .[1]}' build_output/results.json build_output/pyright_output.json > \
    build_output/temp.json && mv build_output/temp.json build_output/results.json

chmod -R 777 .
exit 0 