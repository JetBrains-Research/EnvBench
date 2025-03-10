#!/bin/bash

# Function to activate environment and install dependencies
install_dependencies() {
    # Find all requirements files including those with requirements in the middle
    REQ_FILES=$(find . -maxdepth 1 -type f -name "*requirements*.txt")
    if [ -n "$REQ_FILES" ]; then
        echo "Found requirements files:"
        echo "$REQ_FILES"
        for req_file in $REQ_FILES; do
            echo "Installing from $req_file..."
            python -m pip install -r "$req_file"
        done
    elif [ -f "setup.py" ]; then
        echo "Installing from setup.py..."
        python -m pip install -e .
    elif [ -f "pyproject.toml" ]; then
        echo "Installing from pyproject.toml..."
        python -m pip install -e .
    elif [ -f "setup.cfg" ]; then
        echo "Installing from setup.cfg..."
        python -m pip install -e .
    elif [ -f "Pipfile" ]; then
        echo "Installing from Pipfile..."
        pipenv install --deploy
    else
        echo "No requirements found"
        exit 444
    fi
}

# Try to activate the appropriate Python environment
if [ -f "environment.yml" ]; then
    echo "Detected Conda environment..."
    CONDA_ENV_NAME=$(head -n 1 environment.yml 2>/dev/null || head -n 1 environment.yaml | cut -d' ' -f2)
    conda env create -f environment.yml
    conda activate "$CONDA_ENV_NAME"
    install_dependencies
elif [ -f "uv.lock" ]; then
    echo "Detected uv.lock..."
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
elif [ -f "poetry.lock" ]; then
    echo "Detected poetry.lock..."
    poetry install -n --no-root --all-groups --all-extras
    echo "Activating Poetry environment..."
    # eval "$(poetry env activate -n)" # TODO: this is not working

    # Get Python executable path from Poetry
    POETRY_PYTHON=$(poetry run which python)
    if [ -z "$POETRY_PYTHON" ]; then
        echo "Failed to get Python path from Poetry"
        exit 1
    fi
    # Convert Python path to virtualenv path and activation script
    VENV_PATH=$(dirname $(dirname "$POETRY_PYTHON"))
    ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"

    if [ ! -f "$ACTIVATE_SCRIPT" ]; then
        echo "Poetry virtualenv activate script not found at $ACTIVATE_SCRIPT"
        exit 1
    fi
    echo "Activating Poetry virtualenv at $VENV_PATH"
    source "$ACTIVATE_SCRIPT"
    echo "Poetry environment activated: $(which python)"
elif [ -f "pyproject.toml" ]; then
    echo "Detected pyproject.toml..."
    install_dependencies
elif [ -f "requirements.txt" ] || [ -f "Pipfile" ] || [ -f "setup.py" ]; then
    echo "Detected requirements.txt, Pipfile, or setup.py..."
    install_dependencies
else
    echo "No recognized Python environment or requirements found in the root"
    exit 444
fi
