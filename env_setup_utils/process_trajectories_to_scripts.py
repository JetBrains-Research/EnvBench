import argparse
import json
import logging
import os
import tempfile
from typing import Any, Dict, List

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_tree, upload_file  # type: ignore[import-untyped]
import jsonlines
from tqdm import tqdm  # type: ignore[import-untyped]

load_dotenv()


def parse_script_from_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    """Processes a given trajectory into a final bash script.

    The last message in the trajectory should be a commands_history node containing
    a list of executed commands with their exit codes.

    Example of the last message format:
    {
        "timestamp": "2025-01-02T16:25:48.936260",
        "node": "commands_history",
        "commands": [
            {"command": "ls -R", "exit_code": 0},
            {"command": "cat file.txt", "exit_code": 0}
        ]
    }

    Commands are filtered to exclude:
    - Failed commands (exit_code != 0)
    - Commands starting with: ./gradlew, gradlew, mvn, gradle
    Failed commands are commented out in the output.
    """
    if not trajectory:
        logging.warning("Empty trajectory.")
        return ""

    last_message = trajectory[-1]
    if last_message.get("node") != "commands_history":
        logging.warning("Last message is not a commands_history node.")
        return ""

    def filter_command(command: dict[str, Any]) -> bool:
        if command.get("exit_code") != 0:
            return False
        return True

    def format_command(command: dict[str, Any]) -> str:
        bash_command = command["command"]
        if filter_command(command):
            return bash_command
        else:
            return f"# {bash_command}"

    commands = last_message.get("commands", [])
    if isinstance(commands, str):
        commands = json.loads(commands)

    return "\n".join(format_command(command) for command in commands)


def process_trajectories_to_scripts(trajectories_dataset: str, input_trajectories_dir: str):
    scripts = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for trajectory_file in tqdm(
            list_repo_tree(
                trajectories_dataset, os.path.join(input_trajectories_dir, "trajectories"), repo_type="dataset"
            )
        ):
            file_path = hf_hub_download(
                repo_id=trajectories_dataset,
                filename=trajectory_file.path,
                repo_type="dataset",
                local_dir=temp_dir,
            )

            with jsonlines.open(file_path, "r") as reader:
                trajectory = [line for line in reader]
            repository, revision = os.path.basename(trajectory_file.path[: -len(".jsonl")]).split("@")
            scripts.append(
                {
                    "repository": repository.replace("__", "/"),
                    "revision": revision,
                    "script": parse_script_from_trajectory(trajectory),
                }
            )

        with jsonlines.open(f"{temp_dir}/scripts.jsonl", "w") as writer:
            writer.write_all(scripts)

        upload_file(
            path_in_repo=os.path.join(input_trajectories_dir, "scripts.jsonl"),
            path_or_fileobj=f"{temp_dir}/scripts.jsonl",
            repo_id=trajectories_dataset,
            repo_type="dataset",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-trajectories-dir",
        required=True,
        type=str,
        help="The directory in the HF trajectories dataset that contains the trajectories to be processed.",
    )
    parser.add_argument(
        "--traj_repo_id",
        default="JetBrains-Research/EnvBench-trajectories",
        type=str,
        help="The repository ID of the trajectories dataset.",
    )
    args = parser.parse_args()

    process_trajectories_to_scripts(
        args.traj_repo_id,
        args.input_trajectories_dir,
    )
