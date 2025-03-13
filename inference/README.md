# 🌱⚙️ EnvBench: Inference

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

[uv](https://github.com/astral-sh/uv) is used for dependencies management. To install dependencies, go to root directory and run:

```shell
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

### Configure

[Hydra](https://hydra.cc/docs/intro/) is used for configuration. Modify one of the existing configs ([`run_inference_jvm.yaml`](configs/run_inference_jvm.yaml), [`run_inference_py.yaml`](configs/run_inference_py.yaml)) or create a new one under [`configs`](configs) directory.

For more information about available options, see `EnvSetupRunnerConfig` in [`configs/run_inference_config.py`](configs/run_inference_config.py) and its sub-configs.

### Run

```shell
python run_inference.py --config-name your-config-name
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
