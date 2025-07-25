import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Awaitable, Dict, List, Sequence

from dotenv import load_dotenv
from huggingface_hub import HfApi  # type: ignore[import-untyped]
import hydra
import jsonlines
from omegaconf import DictConfig, OmegaConf

from inference.configs import EnvSetupRunnerConfig
from inference.src.env_setup_runner import EnvSetupRunner

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
    extra_info: Dict[str, Any],
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
            extra_info=extra_info,
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


async def run_experiment(cfg: DictConfig):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    cfg_model = EnvSetupRunnerConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore

    if cfg_model.rewrite_trajectories:
        if os.path.exists(cfg_model.logging_dir):
            shutil.rmtree(cfg_model.logging_dir)

    data_source = getattr(cfg_model.data_source, cfg_model.data_source.type).instantiate()
    if first_n := os.getenv('ONLY_FIRST_N'):
        n = int(first_n)
        logging.info(f"Limiting data source to first {n} examples.")
        from itertools import islice
        data_source = islice(data_source, n)
    os.makedirs(cfg_model.logging_dir, exist_ok=True)

    if not cfg_model.rewrite_trajectories and os.path.exists(cfg_model.logging_dir):
        processed_trajectories: List[Dict[str, str]] = []
        for trajectory_file in os.listdir(cfg_model.logging_dir):
            repository, revision = trajectory_file[: -len(".jsonl")].split("@")
            repository = repository.replace("__", "/")

            with jsonlines.open(os.path.join(cfg_model.logging_dir, trajectory_file)) as reader:
                messages = [line for line in reader]
            if messages and messages[-1]["node"] == "commands_history":
                processed_trajectories.append({"repository": repository, "revision": revision})

        examples_to_process = [
            example
            for example in data_source
            if {"repository": example["repository"], "revision": example["revision"]} not in processed_trajectories
        ]
    else:
        examples_to_process = list(data_source)

    # for debugging: limit to a specified number
    if cfg_model.debug_limit is not None and cfg_model.debug_limit > 0:
        examples_to_process = examples_to_process[: cfg_model.debug_limit]
        logging.info(f"Debug mode: limiting to first {cfg_model.debug_limit} examples")

    coroutines = [
        process_single_datapoint(
            config=cfg_model,
            repository=example["repository"],
            revision=example["revision"],
            extra_info={key: value for key, value in example.items() if key not in ["repository", "revision"]},
        )
        for example in examples_to_process
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
        # hf_api.upload_folder(  # failing to upload_folder
        #     folder_path=cfg_model.logging_dir,
        #     path_in_repo=os.path.join(cfg_model.hf.path_in_repo, "trajectories"),
        #     repo_id=cfg_model.hf.repo_id,
        #     repo_type="dataset",
        # )

        print('Skipping uploading folder to HuggingFace, please use `hf upload` command manually.')
        # import zipfile
        # TODO: fix
        #
        # zip_path = os.path.join(cfg_model.logging_dir, "trajectories.zip")
        # with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        #     for root, _, files in os.walk(cfg_model.logging_dir):
        #         for file in files:
        #             file_path = os.path.join(root, file)
        #             arcname = os.path.relpath(file_path, cfg_model.logging_dir)
        #             zipf.write(file_path, arcname=arcname)
        #
        # hf_api.upload_file(
        #     path_or_fileobj=zip_path,
        #     path_in_repo=os.path.join(cfg_model.hf.path_in_repo, "trajectories.zip"),
        #     repo_id=cfg_model.hf.repo_id,
        #     repo_type="dataset",
        # )

        try:
            config_name = hydra.core.config_store.ConfigSource.config_name
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
        except (ValueError, AttributeError):
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
                "Couldn't access the current commit to upload to "
                f"{os.path.join(cfg_model.hf.path_in_repo, 'commit_hash.txt')}."
            )


@hydra.main(version_base="1.1", config_path="configs", config_name="run_inference_py")
def main(cfg: DictConfig) -> None:
    """Launch Environment Setup experiment using Hydra configuration."""
    asyncio.run(run_experiment(cfg))


if __name__ == "__main__":
    main()
