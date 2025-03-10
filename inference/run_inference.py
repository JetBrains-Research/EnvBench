import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from argparse import ArgumentParser
from typing import Any, Awaitable, Dict, List, Sequence

import jsonlines
from dotenv import load_dotenv
from huggingface_hub import HfApi  # type: ignore[import-untyped]
from hydra import compose, initialize
from omegaconf import OmegaConf

from configs import EnvSetupRunnerConfig
from src.env_setup_runner import EnvSetupRunner

load_dotenv()

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


async def run_limited(coroutines: Sequence[Awaitable[Any]], batch_size: int):
    sem = asyncio.Semaphore(batch_size)

    async def runner(coro):
        async with sem:
            return await coro

    tasks = [asyncio.create_task(runner(c)) for c in coroutines]
    return await asyncio.gather(*tasks)


async def process_single_datapoint(
    repository: str,
    revision: str,
    config: EnvSetupRunnerConfig,
) -> None:
    try:
        toolkit = await config.agent.toolkit.instantiate(
            repository=repository,
            revision=revision,
            image=config.docker.image,
            error_message=config.docker.error_message,
            env_vars=config.docker.env_vars,
            repository_workdir=config.docker.repository_workdir,
            container_start_timeout=config.docker.container_start_timeout,
            bash_timeout=config.docker.bash_timeout,
            max_num_chars_bash_output=config.docker.max_num_chars_bash_output,
            hf_name=config.docker.hf_name,
            output_dir=config.docker.output_dir,
            language=config.docker.language,
            clear_repo=config.docker.clear_repo,
        )

        agent = config.agent.instantiate(toolkit=toolkit)

        runner = EnvSetupRunner(
            repository=repository,
            revision=revision,
            agent=agent,
            log_trajectory=config.log_trajectory,
            logging_dir=config.logging_dir,
        )
        if config.global_timeout:
            try:
                await asyncio.wait_for(runner.arun(), timeout=config.global_timeout)
            except asyncio.TimeoutError:
                logging.warning(
                    f"[{repository}@{revision}] Stopped due to reaching global timeout {config.global_timeout}."
                )
        else:
            await runner.arun()
        try:
            await asyncio.wait_for(toolkit.clean(), timeout=60 * 3)
        except asyncio.TimeoutError:
            logging.warning(f"[{repository}@{revision}] Unable to clean container in 3 minutes.")
        return None

    except Exception:
        logging.error(f"An error occurred for {repository}@{revision}: {traceback.format_exc()}")
        return None


async def main(config_name: str, config_path: str):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    with initialize(version_base="1.1", config_path=config_path):
        cfg = compose(config_name=config_name)

        data_root = os.getenv("DATA_ROOT")
        if data_root is None:
            raise ValueError("Please set an environment variable `DATA_ROOT` to specify where data is stored.")

        cfg.data_source.local.path = os.path.join(data_root, cfg.data_source.local.path)

        cfg_model = EnvSetupRunnerConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore
        if cfg_model.rewrite_trajectories:
            if os.path.exists(cfg_model.logging_dir):
                shutil.rmtree(cfg_model.logging_dir)

        data_source = getattr(cfg_model.data_source, cfg_model.data_source.type).instantiate()

        if not cfg_model.rewrite_trajectories and os.path.exists(cfg_model.logging_dir):
            processed_trajectories: List[Dict[str, str]] = []
            for trajectory_file in os.listdir(cfg_model.logging_dir):
                repository, revision = trajectory_file[: -len(".jsonl")].split("@")
                repository = repository.replace("__", "/")

                with jsonlines.open(os.path.join(cfg_model.logging_dir, trajectory_file)) as reader:
                    messages = [line for line in reader]
                if messages and messages[-1]["node"] == "commands_history":
                    processed_trajectories.append({"repository": repository, "revision": revision})

            coroutines = [
                process_single_datapoint(
                    config=cfg_model,
                    repository=example["repository"],
                    revision=example["revision"],
                )
                for example in data_source
                if {"repository": example["repository"], "revision": example["revision"]} not in processed_trajectories
            ]
        else:
            coroutines = [
                process_single_datapoint(
                    config=cfg_model,
                    repository=example["repository"],
                    revision=example["revision"],
                )
                for example in data_source
            ]

        logging.info(f"Got {len(coroutines)} repositories to process.")

        if cfg_model.langsmith_project is not None:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_PROJECT"] = cfg_model.langsmith_project

        if cfg_model.max_concurrent:
            await run_limited(coroutines, cfg_model.max_concurrent)
        else:
            for task_future in asyncio.as_completed(coroutines):
                await task_future

        if cfg_model.hf.upload:
            hf_api = HfApi()
            hf_api.upload_folder(
                folder_path=cfg_model.logging_dir,
                path_in_repo=os.path.join(cfg_model.hf.path_in_repo, "trajectories"),
                repo_id=cfg_model.hf.repo_id,
                repo_type="dataset",
            )

            try:
                if not config_name.endswith(".yaml"):
                    config_name += ".yaml"

                if os.path.exists(f"configs/{config_name}"):
                    path = f"configs/{config_name}"
                else:
                    path = f"inference/configs/{config_name}"
                hf_api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=os.path.join(cfg_model.hf.path_in_repo, "config.yaml"),
                    repo_id=cfg_model.hf.repo_id,
                    repo_type="dataset",
                )
            except ValueError:
                logging.error(
                    f"Couldn't access the config to upload to {os.path.join(cfg_model.hf.path_in_repo, 'config.yaml')}."
                )

            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = os.path.join(temp_dir, "tempfile.txt")
                    with open(temp_file_path, "w") as temp_file:
                        temp_file.write(subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8"))

                    hf_api.upload_file(
                        path_or_fileobj=temp_file_path,
                        path_in_repo=os.path.join(cfg_model.hf.path_in_repo, "commit_hash.txt"),
                        repo_id=cfg_model.hf.repo_id,
                        repo_type="dataset",
                    )
            except subprocess.CalledProcessError:
                logging.error(
                    f"Couldn't access the current commit to upload to {os.path.join(cfg_model.hf.path_in_repo, 'commit_hash.txt')}."
                )


if __name__ == "__main__":
    parser = ArgumentParser(description="Launch Environment Setup experiment.")
    parser.add_argument("--config-name", type=str, help="Which config under configs directory to use.", required=True)
    parser.add_argument("--config-path", type=str, help="Path to the config file.", default="configs")
    args = parser.parse_args()

    asyncio.run(main(args.config_name, args.config_path))
