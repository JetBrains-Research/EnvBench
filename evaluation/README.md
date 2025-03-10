# EnvSetup Evaluation

> Kudos to Ilia Kholkin; this script was initially implemented by him. This folder contains a slightly modified version of [this GitHub repo](https://github.com/kholkinilia/qodana-eval). 

TODO: rewrite README to reflect the current state of the project

## How to run

1. Start a Docker daemon
2. Create a `.env` file in a repository root specifying `QODANA_TOKEN` and `HF_TOKEN`
3. Modify `conf/config.yaml` to your liking
4. Run `poetry install`
5. Run `poetry run python3 main.py`

## Data format

### Input data

The input file should be in a `JSONLines` format with the following fields:

* `repository`: a GitHub repo name including the author or organisation (e.g. `kholkinilia/qodana-eval`)
* `revision`: a commit_sha of the desired version of the repo (e.g. `cdc69a6608fbebe7ca3de6533bb619cc443b53f2`)

### Output data

#### Jsonl file

It's saved at `data.language.<language type>.result_paths.jsonl` from the config file.
This file contains jsons, one for each repository, of the following structure:

- `repo_name`: a GitHub repo name including the author or organisation
- `commit_sha`: a commit_sha of the desired version of the repo
- `exit_code`: exit code of Qodana or if timeout or unknown errors happened the respective codes from config are used
- `execution_time`: roughly execution time in seconds to run a Qodana container over the repo. Calculated using python
  time library...
- `result_archive_name`: a name of the archive containing Qodana results directory with all of Qodana logs and results

An example for one repo:

```json
{
  "repo_name": "tcurdt/jdeb",
  "commit_sha": "8f4fc5a14b7a116d72fad627c8e7a31d946783b3",
  "exit_code": 0,
  "execution_time": 87.95895314216614,
  "result_archive_name": "tcurdt__jdeb@8f4fc5a14b7a116d72fad627c8e7a31d946783b3.zip"
}
```

#### Archives

The Qodana logs and results archives with names corresponding to those specified in the `.jsonl` results file.
