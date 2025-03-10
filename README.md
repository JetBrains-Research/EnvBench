# 🌱⚙️ EnvBench
A Benchmark for Automated Environment Setup

<p align="center">
  <img src=".github/overview.png" alt="Environment Setup Pipeline Overview" width="800"/>
</p>

## Overview

This project automates the process of setting up development environments by analyzing project requirements and configuring the necessary tools and dependencies. It supports both Python and JVM-based projects.

## Prerequisites

- [Poetry](https://python-poetry.org/) for Python dependency management
- [Docker](https://www.docker.com/) for running isolated environments

## Project Structure

```
env-setup/
├── data_collection/    # Repository analysis and data gathering tools
├── env_setup_utils/    # Core utilities for inference and evaluation
├── evaluation/         # Test suites for Python and JVM environments
└── inference/          # Environment setup agents and Docker environments
```

## Installation

Install dependencies for each component:
```bash
poetry install -C evaluation
poetry install -C inference
poetry install -C env_setup_utils
poetry install -C data_collection
```

## Usage

### Running the Full Pipeline

To run the complete pipeline (inference and evaluation):

```bash
cd env_setup_utils && poetry run python scripts/full_pipeline.py
```

Results are automatically uploaded to the `trajectories` repository on HuggingFace.

### Running Specific Agents

Use Hydra to configure and run specific agents:

```bash
cd env_setup_utils

# Run JVM environment setup
poetry run python scripts/full_pipeline.py -cn jvm

# Run Python environment setup
poetry run python scripts/full_pipeline.py -cn python
```

For all configuration options, see [conf/defaults.yaml](env_setup_utils/scripts/conf/defaults.yaml).

## Documentation

For detailed documentation on each component:

- [Data Collection](data_collection/README.md)
- [Environment Setup Utils](env_setup_utils/README.md)
- [Evaluation](evaluation/README.md)
- [Inference](inference/README.md)

## Implementation Details

- [Agents and Prompts](inference/src/agents)
- [Dockerfiles](env_setup_utils/scripts)
- [Deterministic and Evaluation Scripts](evaluation/scripts)

## Downloads

### Data
- [Dataset](https://huggingface.co/datasets/JetBrains-Research/EnvBench)
- [Trajectories](https://huggingface.co/datasets/JetBrains-Research/EnvBench-trajectories)

## Citation

If you find our work helpful, please use the following citation:

```
@inproceedings{
eliseeva2025envbench,
title={EnvBench: A Benchmark for Automated Environment Setup},
author={Aleksandra Eliseeva and Alexander Kovrigin and Ilia Kholkin and Egor Bogomolov and Yaroslav Zharov},
booktitle={ICLR 2025 Third Workshop on Deep Learning for Code},
year={2025},
url={https://openreview.net/forum?id=izy1oaAOeX}
}
```

## License

MIT. Check `LICENSE`.