# 🌱⚙️ EnvBench
A Benchmark for Automated Environment Setup

<p align="center">
  <img src=".github/overview.png" alt="Environment Setup Pipeline Overview" width="800"/>
</p>

## Overview

This project automates the process of setting up development environments by analyzing project requirements and configuring the necessary tools and dependencies. It supports both Python and JVM-based projects.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) for dependency management
- [Docker](https://www.docker.com/) for running isolated environments

## Running the Benchmark

### Setup

Setup a virtual environment and install dependencies using uv.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

### Running the Pipeline

To run the complete pipeline (inference and evaluation):

```bash
uv run envbench \
    -cn python-bash \
    llm@inference.agent=gpt-4o-mini \
    traj_repo_id=<your-hf-username>/<your-repo-name> \ # repository to save trajectories
    use_wandb=true
```

Results are automatically uploaded to the provided trajectories repository on HuggingFace.

For all configuration options, including different agents and llms, see [conf](conf) directory with Hydra configs.

If you want to run the pipeline only for evaluation, you can use the following command:

```bash
uv run envbench -cn python-bash skip_inference=true skip_processing=true run_name<your-run-name>
```

Alternatively, take a look at the [evaluation/main.py](evaluation/main.py) file for more details on how to run the evaluation step.

## Implementation Details

- [Agents and Prompts](inference/src/agents)
- [Dockerfiles](dockerfiles)
- [Deterministic and Evaluation Scripts](evaluation/scripts)

## Artifacts
- [Dataset](https://huggingface.co/datasets/JetBrains-Research/EnvBench)
- [Trajectories from the paper](https://huggingface.co/datasets/JetBrains-Research/EnvBench-trajectories)

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