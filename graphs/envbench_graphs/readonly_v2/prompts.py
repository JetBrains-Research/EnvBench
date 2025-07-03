from pathlib import Path
from textwrap import dedent
from typing import List

import requests  # type: ignore[import-untyped]

try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files  # type: ignore

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .state_schema import EnvSetupReadOnlyState


def get_dockerfile_content() -> str:
    """Get Dockerfile content from package data, local file, or GitHub fallback."""
    # Try package resources first (when installed as package)
    try:
        dockerfile_text = files("envbench_graphs").joinpath("python.Dockerfile").read_text()
        return dockerfile_text
    except Exception:
        pass

    # Try local file (for development)
    try:
        dockerfile_path = Path(__file__).parents[3] / "dockerfiles" / "python.Dockerfile"
        if dockerfile_path.exists():
            return dockerfile_path.read_text()
    except Exception:
        pass

    # Fallback to GitHub if local file fails
    github_url = "https://raw.githubusercontent.com/JetBrains-Research/EnvBench/main/dockerfiles/python.Dockerfile"

    try:
        response = requests.get(github_url)
        return response.text
    except Exception:
        pass

    # Final fallback - basic description
    return "Ubuntu 22.04 with Python, pyenv, poetry, and development tools installed."


dockerfile = get_dockerfile_content()

system_prompt = dedent(
    """
    You are an intelligent AI agent with the goal of installing and configuring all necessary dependencies for a given Python repository.
    The repository is already located in your working directory, so you can immediately access all files and folders..

    You are operating in a Docker container with Python and the necessary system utilities. 
    For your reference, the Dockerfile used is:

    ```
    {dockerfile}
    ```

    You are provided access to a Bash terminal, which you can use to perform READ-ONLY operations, such as reading files and folders, searching within files, 
    or checking system configuration, to gather the information needed to complete your task.

    Carefully check:
    - Documentation (README.md, CONTRIBUTING.md, etc.) in the repository.
    - All relevant configuration files in the repository (pyproject.toml, requirements.txt, setup.py, etc.).
    
    Specifically, you should pay attention to:
    - Any requirements on the Python version (in the final shell script, you will need to install Python via pyenv if the required version is not available on the system).
    - Any additional system packages required for the repository (in the final shell script, you will need to install them via apt-get).
    - What dependency manager is used in the repository (pip or Poetry).
    - Where the configuration files for installing the dependencies are located.
    - Optional dependency groups and extras in the configuration files (in the final shell script, you will need to install all of them).

    Once you gathered the information you need, submit the final shell script using the following format:

    ```bash
    # Your shell script here
    # Install dependencies, configure environment, etc.
    ```

    Remember:
    - You can only execute READ operations. Any attempt to execute WRITE operations will result in an error.
    - For the final shell script:
      - Pay attention to what tools are already available on the system. You don't need to install pyenv, poetry, conda, or pip.
      - To install Python, use pyenv. Include 'pyenv install -f <version>' and 'pyenv local <version>' in the final shell script.
      - Don't forget to configure the system to use the right Python binary. 
        - If the repository uses Poetry, you will need to include 'pyenv local <version>' and 'poetry env use `which python`' in the final shell script before running `poetry install`.
          After `poetry install`, you will also need to include 'source $(poetry env info --path)/bin/activate' to the final shell script.
        - If the repository uses pip, you will need to include 'pyenv local <version>' in the final shell script before running `python -m pip install`.
    """
).format(dockerfile=dockerfile)


def get_readonly_prompt(state: EnvSetupReadOnlyState, config: RunnableConfig) -> List[BaseMessage]:
    return [SystemMessage(content=system_prompt)]
