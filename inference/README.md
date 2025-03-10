# 🌱⚙️ Environment Setup: Inference

This repository contains the code for running agents on Environment Setup datasets.

## Table of Contents

1. [How-to](#how-to)
   1. [Install dependencies](#install-dependencies)
   2. [Configure](#configure)
   3. [Run](#run)
2. [About](#about)
3. [Demo](#demo)

## How-to

### Install dependencies

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

### Configure

[Hydra](https://hydra.cc/docs/intro/) is used for configuration. Modify one of the existing configs ([`run_inference_jvm.yaml`](configs/run_inference_jvm.yaml), [`run_inference_py.yaml`](configs/run_inference_py.yaml)) or create a new one under [`configs`](configs) directory.

For more information about available options, see `EnvSetupRunnerConfig` in [`configs/run_inference_config.py`](configs/run_inference_config.py) and its sub-configs.

### Run

```shell
poetry run python run_inference.py --config-name your-config-name
```

## About

Also note that the script provides an option to log agents' trajectories and upload them to HuggingFace.

### Agents

> Located under [`src/agents`](src/agents).

We use [LangGraph](https://langchain-ai.github.io/langgraph/) library for the implementations of the agents. The expected interface for an agent is defined in [`BaseEnvSetupAgent`](src/agents/base.py). The current version provides two agents: for [Python](src/agents/python) and for [JVM](src/agents/jvm) languages.

There is also a [Python baseline](src/agents/python_baseline) that is implemented via LangGraph, but features no LLM calls; all logic is hard-coded.

### Toolkits

> Located under [`src/toolkits`](src/toolkits).

The expected interface for a toolkit and utilities to interact with Docker via Bash commands are defined in [`BaseEnvSetupToolkit`](src/toolkits/base.py).

The current version provides one toolkit that allows launching arbitrary Bash commands in a Docker container.

For the implementation, see `BashTerminalToolkit` in [`src/toolkits/bash_terminal.py`](src/toolkits/bash_terminal.py).

### Context providers

> Located under [`src/context_providers`](src/context_providers).

Aside from that, we provide utilities for collecting relevant context. Internally, current options parse data from our HuggingFace datasets.

* For *build instructions*, the available options are located in [`src/context_providers/build_instructions.py`](src/context_providers/build_instructions.py). 
    * As of now, primary option is `SimpleREADMEEnvSetupInstructionProvider`, which parses repositories' READMEs and includes all sections containing build-related keywords.
