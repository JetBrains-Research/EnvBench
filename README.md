# 🌱⚙️ Environment Setup

Repository for **Environment Setup** project.

## Repository structure

Currently, the repository consists of multiple subprojects. Refer to READMEs under each folder for more details

* Data Collection: available under [`data_collection`](data_collection/README.md) folder.
* EDA
  * Python: available under [`eda/python`](eda/python/README.md) folder.
* Utils: available under [`env_setup_utils`](env_setup_utils/README.md) folder.
* Inference: available under [`inference`](inference/README.md) folder.
* Evaluation: available under [`qodana-eval`](qodana-eval/README.md) folder. **TODO**

## Experiments 101

> **TODO**

All the results from our experiments are available in 🤗 [`JetBrains-Research/ai-agents-env-setup-trajectories`](https://huggingface.co/datasets/JetBrains-Research/ai-agents-env-setup-trajectories) dataset (private! you need to join the org!).

[Here](https://huggingface.co/datasets/JetBrains-Research/ai-agents-env-setup-trajectories/tree/main/python_baseline) is an example of the structure of the folder for one of the experiments:

```
/experiment_folder
├── qodana_archives/
├── trajectories/
├── commit_hash.txt
├── config.yaml
├── results.jsonl
└── scripts.jsonl
```

### Step 1: obtain agent trajectories

The first thing you need to do is gather trajectories from one of our agents. This step gets you [`trajectories`](https://huggingface.co/datasets/JetBrains-Research/ai-agents-env-setup-trajectories/tree/main/python_baseline/trajectories) directory that contains the trajectories an agent took for each repository as well as [`config.yaml`](https://huggingface.co/datasets/JetBrains-Research/ai-agents-env-setup-trajectories/blob/main/python_baseline/config.yaml) and [`commit_hash.txt`](https://huggingface.co/datasets/JetBrains-Research/ai-agents-env-setup-trajectories/blob/main/python_baseline/commit_hash.txt) to indicate the agent configuration used and the state of the repository at the moment of launching an experiment.

To do this, you need to run `run_inference.py` from `inference` folder in this repo; refer to corresponding [README](ideformer_client/README.md) for further details.

### Step 2: obtain Bash scripts from trajectories

Next, you need to get a Bash script from each trajectory. This step gets you [`scripts.jsonl`](https://huggingface.co/datasets/JetBrains-Research/ai-agents-env-setup-trajectories/blob/main/python_baseline/scripts.jsonl) file.

The preferrable way to achieve this is to run [`process_trajectories_to_scripts.py`](https://jetbrains.team/p/ml-4-se-lab/repositories/ai-agents-env-setup/files/env_setup_utils/env_setup_utils/process_trajectories_to_scripts.py) script from `env-setup-utils` folder. 

### Step 3: obtain evaluation results

Next, you need to launch evaluation on each repository with the scripts obtained on the previous step. This step gets you [`qodana_archives`](https://huggingface.co/datasets/JetBrains-Research/ai-agents-env-setup-trajectories/tree/main/python_baseline/qodana_archives) directory with archives with Qodana outputs for each repository and [`results.jsonl`](https://huggingface.co/datasets/JetBrains-Research/ai-agents-env-setup-trajectories/blob/main/python_baseline/results.jsonl) file indicating other details like execution time and exit codes.

To do this, refer to [README](qodana-eval/README.md) from `qodana-eval` folder.

### Step 4: compute metrics from Qodana archives

Finally, you can use the obtained Qodana results to get metrics on how well the agent performed! For this, we have an 🤗 [Evaluate](https://huggingface.co/docs/evaluate/en/index) metric. 

Refer to 🤗 [`JetBrains-Research/qodana_pass`](https://huggingface.co/spaces/JetBrains-Research/qodana_pass) for further details.