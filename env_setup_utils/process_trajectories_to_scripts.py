import argparse
import json
import logging
import os
import tempfile
from typing import Any, Dict, List

import jsonlines
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_tree, upload_file  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

load_dotenv()


def parse_installamatic_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    """
    Extract the shell script from a trajectory (list of JSON objects) produced
    by Installamatic. The function looks for entries that contain a 'shell_script'
    key. If multiple entries are found, it first returns the one from the node
    'generate_shell_script' (if present), and otherwise returns the last one found.

    Args:
        trajectory: A list of dictionaries parsed from a jsonlines file.

    Returns:
        A string containing the shell script.

    Raises:
        ValueError: If no shell_script is found in any entry.
    """
    shell_entries = [entry for entry in trajectory if "shell_script" in entry]

    if not shell_entries:
        logging.warning("No shell_script found in the trajectory.")
        return ""

    for entry in shell_entries:
        if entry.get("node") == "generate_shell_script":
            return entry["shell_script"]

    return shell_entries[-1]["shell_script"]


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
    commands = [command for command in commands if command is not None]

    # if we have *scripts*, we return the last one
    if all(command["command"].startswith("#!/bin/bash\n\n") for command in commands):
        if len(commands) > 0:
            return commands[-1]["command"]
        return ""

    return "\n".join(format_command(command) for command in commands)


def process_trajectory_file(file_path: str) -> Dict[str, str]:
    """Process a single trajectory file and return the script data."""
    with jsonlines.open(file_path, "r") as reader:
        trajectory = [line for line in reader]
    
    # Extract repository and revision from filename
    filename = os.path.basename(file_path)
    if filename.endswith('.jsonl'):
        filename = filename[:-6]  # Remove .jsonl extension
    
    if '@' in filename:
        repository, revision = filename.split('@', 1)
    else:
        logging.warning(f"Could not parse repository and revision from filename: {filename}")
        repository, revision = filename, "unknown"
    
    script = parse_script_from_trajectory(trajectory)
    if not script:
        script = parse_installamatic_trajectory(trajectory)

    return {
        "repository": repository.replace("__", "/"),
        "revision": revision,
        "script": script,
    }


def process_trajectories_to_scripts(
    trajectories_dataset: str, 
    input_trajectories_dir: str, 
    local_path: str | None = None
):
    scripts = []
    with tempfile.TemporaryDirectory() as temp_dir:
    
        if local_path is not None:
            # Handle local path - treat input_trajectories_dir as the local path
            trajectories_path = local_path
            if not os.path.exists(trajectories_path):
                raise FileNotFoundError(f"Trajectories directory not found: {trajectories_path}")

            # Get all .jsonl files in the trajectories directory
            trajectory_files = []
            for root, dirs, files in os.walk(trajectories_path):
                for file in files:
                    if file.endswith('.jsonl'):
                        trajectory_files.append(os.path.join(root, file))

            for file_path in tqdm(trajectory_files, desc="Processing local trajectories"):
                scripts.append(process_trajectory_file(file_path))
        else:
            # Handle remote HuggingFace dataset
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
                scripts.append(process_trajectory_file(file_path))

        # Save to temp file for upload
        upload_file_path = os.path.join(temp_dir, "scripts.jsonl")
        with jsonlines.open(upload_file_path, "w") as writer:
            writer.write_all(scripts)
        # Common upload logic
        upload_file(
            path_in_repo=os.path.join(input_trajectories_dir, "scripts.jsonl"),
            path_or_fileobj=upload_file_path,
            repo_id=trajectories_dataset,
            repo_type="dataset",
        )
        logging.info(f"Scripts uploaded to HuggingFace dataset: {trajectories_dataset}")


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
    parser.add_argument(
        "--local-path",
        type=str,
        help="Local path to the trajectories directory. If provided, will process local files instead of downloading from HuggingFace.",
    )
    args = parser.parse_args()

    process_trajectories_to_scripts(
        args.traj_repo_id,
        args.input_trajectories_dir,
        args.local_path,
    )
