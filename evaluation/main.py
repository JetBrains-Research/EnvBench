import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import time
import aiohttp
from typing import Optional

import aiodocker
from aiodocker.exceptions import DockerError
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, upload_file  # type: ignore[import-untyped]
import hydra
from hydra.utils import to_absolute_path
import jsonlines  # type: ignore[import-untyped]
from omegaconf import DictConfig
import pandas as pd  # type: ignore[import-untyped]

from env_setup_utils.repo_downloader import RepoDownloader

load_dotenv()

class ScriptExceptionError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Error executing agent script: {self.message}"


def read_script(script_name: str) -> str:
    """Read a bash script from the scripts directory."""
    script_path = Path(__file__).parent / "scripts" / script_name
    with open(script_path, "r") as f:
        return f.read()


def remove_bad_commands(script: str) -> str:
    lines = script.split("\n")
    res = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("mvn compile") or stripped_line.startswith("./mvnw compile"):
            continue
        if stripped_line.startswith("mvn test") or stripped_line.startswith("./mvnw test"):
            continue
        if stripped_line.startswith("gradle build") or stripped_line.startswith("./gradlew build"):
            continue
        if stripped_line.startswith("gradle test") or stripped_line.startswith("./gradlew test"):
            continue
        res.append(line)
    return "\n".join(res)


async def _pull_image(image: str):
    docker_client = aiodocker.Docker()
    try:
        await docker_client.images.inspect(image)
        logging.info(f"Image '{image}' already exists locally.")
    except DockerError as e:
        if e.status == 404:
            logging.info(f"Pulling image '{image}'...")
            await docker_client.images.pull(image)
            logging.info(f"Image '{image}' pulled successfully.")
        else:
            logging.error(f"Error retrieving image '{image}': {e}")
            raise


async def run_opensource(
    repo_downloader: RepoDownloader,
    repo_name: str,
    commit_sha: str,
    cfg: DictConfig,
    bootstrap_script: Optional[str] = None,
):
    json_result = {
        "exit_code": None,
        "execution_time": 0.0,
        "repo_name": repo_name,
        "commit_sha": commit_sha,
        "container_logs": None,
        "issues_count": 0,
    }

    logging.info(f"Processing {repo_name}@{commit_sha}")

    if bootstrap_script is None:
        bootstrap_script = (
            read_script("python_baseline.sh") if cfg.language in ["python", "repo2run"] else read_script("jvm_baseline.sh")
        )
        logging.info(f"Using default bootstrap script for {cfg.language}")

    bootstrap_script = remove_bad_commands(bootstrap_script)
    logging.info("removed maven/gradle build commands")

    # Prepare a repository
    logging.info(f"Downloading repository {repo_name}")
    is_downloaded = repo_downloader.download(repo_name, commit_sha)
    if not is_downloaded:
        logging.error(f"Failed to download repository {repo_name}")
        json_result["exit_code"] = cfg.exit_codes.download_failure
        return json_result

    repo_path = repo_downloader.get_repo_dir_path(repo_name, commit_sha)
    logging.info(f"Repository downloaded to {repo_path}")

    # Select appropriate build script based on language
    build_script_functions = {
        "jvm": lambda: read_script("jvm_build.sh"),
        "python": lambda: read_script("python_build.sh"),
        "repo2run": lambda: read_script("repo2run.sh"),
    }

    if cfg.language not in build_script_functions:
        error_msg = (
            f"Unsupported language: {cfg.language}. Supported languages are: {list(build_script_functions.keys())}"
        )
        logging.error(error_msg)
        raise ValueError(error_msg)

    logging.info(f"Using {cfg.language} build script")
    build_script_content = build_script_functions[cfg.language]()

    # Write the build script
    build_script_path = os.path.join(repo_path, "build.sh")
    with open(build_script_path, "w") as f:
        f.write(build_script_content)
    os.chmod(
        build_script_path,
        os.stat(build_script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )
    logging.info("Build script created and made executable")

    # If a bootstrap script is provided, add it to the repository
    if bootstrap_script:
        logging.info("Adding bootstrap script")
        with open(os.path.join(repo_path, "bootstrap_script.sh"), "w") as f:
            f.write(bootstrap_script)
        os.chmod(
            os.path.join(repo_path, "bootstrap_script.sh"),
            os.stat(os.path.join(repo_path, "bootstrap_script.sh")).st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )

    # Setup a docker client
    logging.info("Setting up Docker client")
    docker_client = aiodocker.Docker(session=aiohttp.ClientSession(
        connector=aiohttp.UnixConnector(os.environ.get("DOCKER_HOST", '/var/run/docker.sock')),
        timeout=aiohttp.ClientTimeout(total=cfg.docker.container_timeout,
                                      sock_connect=cfg.docker.create_container_timeout))
    )

    # Prepare container config
    container_config = {
        "Image": cfg.docker.image[cfg.language],
        "Cmd": ["/bin/bash", "-c", "/data/project/build.sh"],
        "WorkingDir": "/data/project",
        "HostConfig": {
            "Binds": [f"{os.path.abspath(repo_path)}:/data/project:rw"],
        },
    }

    start_time = time.perf_counter()
    created_time = start_time
    container = None

    try:
        # Run the build script inside a Docker container with asyncio timeout
        logging.info(f"Starting Docker container for {repo_name}")

        # Measure how long container creation takes
        container = await docker_client.containers.create(config=container_config)
        created_time = time.perf_counter()
        create_duration = created_time - start_time
        logging.info(f"Container created: {container.id[:12]} (took {create_duration:.2f}s)")

        await container.start()
        logging.info(f"Container {container.id[:12]} started in {time.perf_counter() - start_time:.2f}s")

        # Define the part that should be time-limited
        async def stream_and_wait_container():
            nonlocal container

            # logs_buffer = []
            # async for log_chunk in container.log(stdout=True, stderr=True, follow=True):
            #     log_line = log_chunk.strip()
            #     if log_line:
            #         decoded_line = log_line if isinstance(log_line, str) else log_line.decode("utf-8")
            #         logging.info(f"[Docker] {decoded_line}")
            #         logs_buffer.append(decoded_line)

            # Wait for the container to finish
            result = await container.wait(timeout=cfg.docker.container_timeout)
            exit_code = result.get("StatusCode", 1)

            # Collect final logs
            all_logs = await container.log(stdout=True, stderr=True, follow=False)
            container_logs = "".join(log.decode("utf-8") if isinstance(log, bytes) else log for log in all_logs)

            return exit_code, container_logs

        # Apply timeout only to execution phase
        exit_code, container_logs = await asyncio.wait_for(
            stream_and_wait_container(),
            timeout=cfg.docker.container_timeout,
        )

        logging.info(f"Container finished with exit code {exit_code} in {time.perf_counter() - created_time:.2f} seconds")
        json_result["exit_code"] = exit_code
        json_result["container_logs"] = container_logs

        # Try to read the results file from the container
        try:
            results_path = os.path.join(repo_path, "build_output", "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    build_results = json.load(f)
                    for key in build_results:
                        json_result[key] = build_results[key]
                    logging.info(f"Found {json_result.get('issues_count', '??')} issues")
                if cfg.language == "repo2run":
                    json_result["issues_count"] = sum(1 for _ in json_result.get("pytest", {}).get("collectors", []) if _.get("outcome") == "failed")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Error reading results.json: {str(e)}")

    except asyncio.TimeoutError as e:
        logging.error(f"Timeout error: {str(e)}")
        logging.error(f"Container timeout ({cfg.docker.container_timeout}s) reached (actual duration {time.perf_counter() - created_time:.2f}s)")
        json_result["exit_code"] = cfg.exit_codes.timeout
        json_result["container_logs"] = f"Container execution timeout after {cfg.docker.container_timeout} seconds"
    except DockerError as e:
        logging.error(f"Docker error: {str(e)}")
        json_result["exit_code"] = cfg.exit_codes.docker_failure
        json_result["container_logs"] = str(e)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        json_result["exit_code"] = cfg.exit_codes.unknown_failure
        json_result["container_logs"] = f"An unknown exception occurred: {str(e)}"
    finally:
        # Clean up container
        if container:
            try:
                await container.delete(force=True)
                logging.info("Container removed")
            except Exception as e:
                logging.warning(f"Failed to remove container: {str(e)}")

        # Close Docker client
        await docker_client.close()

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        json_result["execution_time"] = execution_time
        logging.info(f"Total execution time: {execution_time:.2f} seconds")

    # Clear temporary repo data
    logging.info("Cleaning up repository data")
    repo_downloader.clear_repo(repo_name, commit_sha)

    json_path = os.path.join(
        to_absolute_path(cfg.operation.dirs.json_results),
        "results",
        f"{repo_name.replace('/', '__')}.json",
    )

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(json_result, f)
    logging.info("Completed")

    return None


async def run_batch_async(
    repo_downloader: RepoDownloader,
    repo_names: list[str],
    commit_shas: list[str],
    cfg: DictConfig,
    scripts: Optional[list[str]] = None,
):
    """Run multiple repositories concurrently with asyncio."""
    semaphore = asyncio.Semaphore(cfg.operation.pool_config.get('max_workers', 4))

    async def run_single_repo(repo_name: str, commit_sha: str, script: Optional[str] = None):
        async with semaphore:
            return await run_opensource(repo_downloader, repo_name, commit_sha, cfg, script)

    tasks = []
    for i, (repo_name, commit_sha) in enumerate(zip(repo_names, commit_shas)):
        script = scripts[i] if scripts else None
        task = run_single_repo(repo_name, commit_sha, script)
        tasks.append(task)

    # Run all tasks concurrently
    await asyncio.gather(*tasks, return_exceptions=True)


# Dictionary of available evaluation tools
eval_tools = {
    "opensource": run_batch_async,
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Draw the repos names&revisions to run a script on
    if cfg.input.mode == "local":
        repos = pd.read_json(to_absolute_path(cfg.input.local), orient="records", lines=True)
    elif cfg.input.mode == "hf":
        local_path = hf_hub_download(
            repo_id=cfg.input.hf.repo_id,
            repo_type="dataset",
            filename=cfg.input.hf.path_in_repo,
        )
        repos = pd.read_json(to_absolute_path(local_path), orient="records", lines=True)
    else:
        raise ValueError("Unknown input source; supported are: 'local' and 'hf'.")
    logging.info(f"Got {len(repos)} repos to process.")

    repo_name_col = cfg.input.columns.repo_name
    commit_sha_col = cfg.input.columns.commit_sha
    script_col = cfg.input.columns.script

    # # only projectsyn/commodore
    # repos = repos[repos[repo_name_col].str.startswith("aqlaboratory/openfold")]
    # repos = repos.head(100)

    assert repo_name_col in repos.columns, (
        f"The input data is expected to have column {repo_name_col} with repository names, but it doesn't."
    )
    assert commit_sha_col in repos.columns, (
        f"The input data is expected to have column {commit_sha_col} with revisions, but it doesn't."
    )
    if cfg.input.use_scripts:
        assert script_col in repos.columns, (
            f"The input data is expected to have column {script_col} with scripts, but it doesn't."
        )

    if not cfg.operation.rewrite_results:
        logging.info("Configured not to overwrite existing results.")
        if os.path.exists(os.path.join(cfg.operation.dirs.json_results, "results")):
            processed_repos = []
            for file in os.listdir(os.path.join(cfg.operation.dirs.json_results, "results")):
                repo_name = file[: -len(".json")].replace("__", "/")
                processed_repos.append({repo_name_col: repo_name})
            processed_repos_df = pd.DataFrame(processed_repos)
            logging.info(f"Got {len(processed_repos_df)} already processed repos.")
            repos = repos.loc[~repos[repo_name_col].isin(processed_repos_df[repo_name_col])]
            logging.info(f"Got {len(repos)} repos to process.")
    else:
        results_dir = os.path.join(cfg.operation.dirs.json_results, 'results')
        logging.info(f"Configured to overwrite existing results. Removing {results_dir}.")
        if os.path.exists(os.path.join(cfg.operation.dirs.json_results, "results")):
            shutil.rmtree(os.path.join(cfg.operation.dirs.json_results, "results"))

    # Create tmp dirs for operation
    os.makedirs(to_absolute_path(cfg.operation.dirs.repo_data), exist_ok=True)
    os.makedirs(to_absolute_path(cfg.operation.dirs.json_results), exist_ok=True)

    # Setup utils class for cloning repositories
    repo_downloader = RepoDownloader(
        hf_name=cfg.input.repos_archives.repo_id,
        output_dir=to_absolute_path(cfg.operation.dirs.repo_data),
        language=cfg.language,
    )

    # Select evaluation tool
    if cfg.eval_tool not in eval_tools:
        raise ValueError(f"Unknown evaluation tool: {cfg.eval_tool}. Supported tools are: {list(eval_tools.keys())}")

    # Pull docker image
    asyncio.run(_pull_image(cfg.docker.image[cfg.language]))

    # Run async batch processing
    func = eval_tools[cfg.eval_tool]
    if cfg.input.use_scripts:
        asyncio.run(func(
            repo_downloader,
            repos[repo_name_col].to_list(),
            repos[commit_sha_col].to_list(),
            cfg,
            repos[script_col].to_list(),
        ))
    else:
        asyncio.run(func(
            repo_downloader,
            repos[repo_name_col].to_list(),
            repos[commit_sha_col].to_list(),
            cfg,
        ))

    # Create local jsonl file with results
    jsonl_path = os.path.join(
        to_absolute_path(cfg.operation.dirs.json_results),
        "results.jsonl",
    )
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    with jsonlines.open(jsonl_path, "w") as writer:
        for file in os.listdir(os.path.join(to_absolute_path(cfg.operation.dirs.json_results), "results")):
            with open(
                os.path.join(to_absolute_path(cfg.operation.dirs.json_results), "results", file),
                "r",
            ) as f:
                json_result = json.load(f)
                writer.write(json_result)

    # Save to huggingface if necessary
    if cfg.output.mode == "hf":
        # Upload results
        upload_file(
            path_or_fileobj=jsonl_path,
            path_in_repo=os.path.join(cfg.output.hf.path_in_repo, "results.jsonl"),
            repo_id=cfg.output.hf.repo_id,
            repo_type="dataset",
        )

        # Remove jsonl if required
        if not cfg.output.keep_local_jsonl:
            os.remove(jsonl_path)


if __name__ == "__main__":
    main()
