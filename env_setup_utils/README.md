# 🌱⚙️ Environment Setup: Utilities

This folder contains various utilities.

## How-to

### Installation

[Poetry](https://python-poetry.org/) is used for dependencies management. To install dependencies, go to root directory and run:

```shell
poetry install
```

**Note**. You can use [`pyenv`](https://github.com/pyenv/pyenv) for installing and managing Python versions.

1. To install a specific Python version (e.g., `3.11.8`) with `pyenv`, run:
 
    ```shell
    pyenv install 3.11.8
    ```
    
    You can also check the already installed versions via:
    
    ```shell
    pyenv versions
    ```

2. Select a specific version for the current shell session: 

    ```shell
    pyenv shell 3.11.8
    ```
    
    Check that Python version was selected correctly via:
    
    ```shell
    python --version
    ```

3. Configure Poetry to use a specific Python executable.

    ```shell
    poetry env use `which python`
    ```
    
    All set! Proceed to installation.

## About

A utility package for Environment Setup. Contains:

* script for processing agent trajectories from [`inference`](../inference) into scripts: [`process_trajectories_to_scripts.py`](env_setup_utils/process_trajectories_to_scripts.py)
* script for parsing Markdown documents based on headings: [`markdown/parse_md_headings.py`](env_setup_utils/markdown/parse_md_headings.py)
* classes for iterating over either local data or data stored on HuggingFace: [`data_sources`](env_setup_utils/data_sources)
* class for downloading repositories either from HuggingFace or GitHub: [`repo_downloader.py`](env_setup_utils/repo_downloader.py)
* script for vizualizing agent trajectories from [`inference`](../inference) as HTML: [`traj2html.py`](env_setup_utils/traj2html.py)
* script for summarizing/analyzing agent trajectories from [`inference`](../inference): [`log_analyzer.py`](env_setup_utils/log_analyzer.py)
