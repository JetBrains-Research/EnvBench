# EnvSetup Evaluation
This folder contains the code for evaluating the environment setup outputs.

> Kudos to [Ilia Kholkin](https://github.com/kholkinilia); this script was initially implemented by him. This folder contains a slightly modified version of [this GitHub repo](https://github.com/kholkinilia/qodana-eval). 

## How to run

```bash
poetry run python main.py
```

See [conf/config.yaml](conf/config.yaml) for the available options.

By default, the results will be uploaded to HuggingFace in the `trajectories` repository.

To run the evaluation with deterministic scripts, set `use_scripts: false` in the config. You can see their code in [scripts](scripts) folder.

## Evaluation scripts
Currently, we have two evaluation scripts:
- [JVM evaluation](scripts/jvm_build.sh)
- [Python evaluation](scripts/python_build.sh)

Their output is `build_output/results.json` file that contains the issues count and auxiliary information. Exit code of the build script is also saved to the results file.