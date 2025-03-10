# 🌱⚙️ Environment Setup: Data Collection

This folder contains the code for data collection (i.e., cloning repositories).

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

The main script for data collection is [`collect_gh_repos.py`](scripts/collect_gh_repos.py).

Refer to `RepoDataCollectionConfig` from [`collect_gh_repos.py`](scripts/collect_gh_repos.py) to tailor the configuration in [collect_gh_repos.yaml](configs/collect_gh_repos.yaml) to your needs, and then run:

```shell
poetry run python scripts/collect_gh_repos.py
```
