#!/usr/bin/env python3

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from analysis.utils import get_dir_path, get_file_path

# Model pricing per 1M tokens (placeholder values)
MODEL_PRICING = {
    "gpt-4o": {"input": 2.5, "output": 10.0},  # $2.50 per 1M input tokens, $10.00 per 1M output tokens
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # $0.15 per 1M input tokens, $0.60 per 1M output tokens
}

# Exit code mapping
EXIT_CODE_MAP = {
    -127: "TIMEOUT",
    -999: "UNKNOWN_FAILURE",
    -888: "DOCKER_FAILURE",
    -777: "CREATE_CONTAINER_FAILURE",
    -666: "DOWNLOAD_FAILURE",
    -555: "SCRIPT_FAILURE",
}


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return results


def load_trajectories(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load all trajectory files from a directory."""
    trajectories = {}
    for file_path in Path(directory).glob("*.jsonl"):
        # Parse repo name and revision from filename
        match = re.match(r"(.+)@([^@]+)\.jsonl", file_path.name)
        if not match:
            continue
        trajectories[file_path.name] = load_jsonl(str(file_path))
    return trajectories


def calculate_message_cost(message: Dict[str, Any]) -> Dict[str, float]:
    """Calculate the cost and token counts for a single message."""
    if (
        message.get("node") != "agent"
        or not message.get("messages")
        or not message["messages"][0].get("message_content", {}).get("usage_metadata")
    ):
        return {"cost": 0.0, "input_tokens": 0, "output_tokens": 0}

    metadata = message["messages"][0]["message_content"]["usage_metadata"]
    model_name = message["messages"][0].get("response_metadata", {}).get("model_name", "")

    # Determine pricing based on model
    if "gpt-4o-mini" in model_name:
        pricing = MODEL_PRICING["gpt-4o-mini"]
    else:
        pricing = MODEL_PRICING["gpt-4o"]

    input_tokens = metadata.get("input_tokens", 0)
    output_tokens = metadata.get("output_tokens", 0)

    # Calculate cost using per 1M token pricing
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    return {"cost": cost, "input_tokens": input_tokens, "output_tokens": output_tokens}


def analyze_trajectory(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze a trajectory to extract key information."""
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    # Calculate total cost and tokens
    for message in messages:
        cost_info = calculate_message_cost(message)
        total_cost += cost_info["cost"]
        total_input_tokens += cost_info["input_tokens"]
        total_output_tokens += cost_info["output_tokens"]

    return {
        "message_count": len(messages),
        "total_cost": round(total_cost, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }


def analyze_trajectories(
    traj_dirs: Union[str, List[str]], repo_id: Optional[str] = None, no_cache: bool = False
) -> pd.DataFrame:
    """Analyze multiple trajectory directories and return results as a DataFrame.

    Args:
        traj_dirs: Directory path(s) containing trajectory files
        repo_id: Optional Hugging Face repo ID to download from
        no_cache: Whether to bypass cache when downloading

    Returns:
        DataFrame with trajectory analysis results
    """
    if isinstance(traj_dirs, str):
        traj_dirs = [traj_dirs]

    results = []
    for traj_dir in traj_dirs:
        # Get directory path (downloading if needed)
        dir_path = get_dir_path(traj_dir, repo_id=repo_id, no_cache=no_cache)

        # Load and analyze trajectories
        trajectories = load_trajectories(dir_path)

        # Calculate totals
        total_cost = 0.0
        total_tokens = 0
        total_messages = 0

        for messages in trajectories.values():
            analysis = analyze_trajectory(messages)
            total_cost += analysis["total_cost"]
            total_tokens += analysis["total_tokens"]
            total_messages += analysis["message_count"]

        # Add results
        results.append(
            {
                "run": traj_dir,
                "n_trajectories": len(trajectories),
                "total_cost": round(total_cost, 4),
                "avg_cost": round(total_cost / len(trajectories), 4) if trajectories else 0,
                "total_tokens": total_tokens,
                "avg_tokens": round(total_tokens / len(trajectories)) if trajectories else 0,
                "total_messages": total_messages,
                "avg_messages": round(total_messages / len(trajectories)) if trajectories else 0,
            }
        )

    return pd.DataFrame(results).set_index("run")


def extract_missing_packages(diagnostics: List[Dict[str, Any]]) -> Set[str]:
    """Extract unique missing packages from diagnostics messages."""
    missing_packages = set()
    pattern = r'Import "([^."]+)'

    for diag in diagnostics:
        if diag.get("rule") == "reportMissingImports":
            if match := re.search(pattern, diag.get("message", "")):
                missing_packages.add(match.group(1))

    return missing_packages


def calculate_exit_code_stats(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate exit code statistics for a single run."""
    return {
        "success": sum(1 for r in results if r["exit_code"] == 0),
        "positive_exit": sum(1 for r in results if r["exit_code"] > 0),
        "negative_exit": sum(1 for r in results if r["exit_code"] < 0),
        **{
            f"exit_{EXIT_CODE_MAP.get(code, str(code))}": count
            for code, count in pd.Series([r["exit_code"] for r in results]).value_counts().items()
        },
    }


def calculate_python_stats(results: List[Dict[str, Any]], success_only: bool = True) -> Dict[str, Any]:
    """Calculate Python-specific statistics."""
    # Calculate pass rates first (using all results)
    total = len(results)
    success_rate = sum(1 for r in results if r["exit_code"] == 0) / total if total > 0 else 0

    # Filter for success_only if needed
    if success_only:
        results = [r for r in results if r["exit_code"] == 0]

    if not results:
        return {
            "total": total,
            "pass_rate": round(success_rate * 100, 2),
            "clean_pass_rate": 0.0,
            "total_missing_imports": 0,
            "avg_missing_imports": 0,
            "total_missing_packages": 0,
            "avg_missing_packages": 0,
        }

    # Calculate missing imports/packages
    total_missing_imports = sum(r.get("issues_count", 0) for r in results)
    total_missing_packages = sum(
        len(extract_missing_packages((r.get("pyright") or {}).get("generalDiagnostics", []))) for r in results
    )

    # Calculate clean pass rate (success AND no missing imports)
    clean_passes = sum(1 for r in results if r.get("issues_count", 0) == 0)
    clean_pass_rate = clean_passes / total if total > 0 else 0

    return {
        "total": total,
        "pass_rate": round(success_rate * 100, 2),
        "clean_pass_rate": round(clean_pass_rate * 100, 2),
        "total_missing_imports": total_missing_imports,
        "avg_missing_imports": round(total_missing_imports / len(results), 2),
        "total_missing_packages": total_missing_packages,
        "avg_missing_packages": round(total_missing_packages / len(results), 2),
    }


def analyze_python_logs(
    logs_files: Union[str, List[str]],
    baseline_file: Optional[str] = None,
    repo_id: Optional[str] = None,
    no_cache: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze Python logs files and return statistics as DataFrames.

    Args:
        logs_files: Path(s) to the JSONL logs file(s)
        baseline_file: Optional path to baseline JSONL file to compare against
        repo_id: Optional Hugging Face repo ID to download from
        no_cache: Whether to bypass cache when downloading

    Returns:
        Tuple of DataFrames:
        - Per-run statistics including baseline (index=run)
        - Comparison with baseline (index=run minus baseline, columns=run_df columns)
    """
    if isinstance(logs_files, str):
        logs_files = [logs_files]

    # Add baseline to logs_files if provided
    if baseline_file:
        logs_files = [*logs_files, baseline_file]

    # Load and analyze all runs
    run_stats = []

    for logs_file in logs_files:
        # Get file path (downloading if needed)
        try:
            file_path = get_file_path(logs_file, caller_name="view_logs", repo_id=repo_id, no_cache=no_cache)
        except:  # noqa: E722
            file_path = get_file_path(
                logs_file,
                caller_name="view_logs",
                repo_id="JetBrains-Research/EnvBench-trajectories",
                no_cache=no_cache,
            )

        results = load_jsonl(file_path)

        # Verify it's a Python run
        if not any(r.get("pyright") is not None for r in results):
            continue

        # Calculate Python stats
        stats = calculate_python_stats(results)
        stats["run"] = logs_file
        run_stats.append(stats)

    # Create run DataFrame
    run_df = pd.DataFrame(run_stats).set_index("run")

    # Handle baseline comparison
    comparison = pd.DataFrame()
    if baseline_file and baseline_file in run_df.index:
        # Get baseline stats
        baseline_stats = run_df.loc[baseline_file]

        # Calculate differences for all other runs
        other_runs = run_df.index != baseline_file
        comparison = pd.DataFrame({col: run_df.loc[other_runs, col] - baseline_stats[col] for col in run_df.columns})

    return run_df, comparison


def analyze_exit_codes(
    logs_files: Union[str, List[str]],
    baseline_file: Optional[str] = None,
    repo_id: Optional[str] = None,
    no_cache: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze exit codes from multiple log files.

    Args:
        logs_files: Path(s) to the JSONL logs file(s)
        baseline_file: Optional path to baseline JSONL file to compare against
        repo_id: Optional Hugging Face repo ID to download from
        no_cache: Whether to bypass cache when downloading

    Returns:
        Tuple of DataFrames:
        - Per-run exit code statistics including baseline (index=run)
        - Comparison with baseline (index=run minus baseline, columns=exit_df columns)
    """
    if isinstance(logs_files, str):
        logs_files = [logs_files]

    # Add baseline to logs_files if provided
    if baseline_file:
        logs_files = [*logs_files, baseline_file]

    # Load and analyze all runs
    run_stats = []

    for logs_file in logs_files:
        # Get file path (downloading if needed)
        try:
            file_path = get_file_path(logs_file, caller_name="view_logs", repo_id=repo_id, no_cache=no_cache)
        except:  # noqa: E722
            file_path = get_file_path(
                logs_file,
                caller_name="view_logs",
                repo_id="JetBrains-Research/EnvBench-trajectories",
                no_cache=no_cache,
            )

        results = load_jsonl(file_path)

        # Calculate exit code stats
        stats = calculate_exit_code_stats(results)

        # Calculate pass rate
        total = len(results)
        stats.update(
            {
                "run": logs_file,
                "total": total,
                "pass_rate": round(100 * stats["success"] / total, 2) if total > 0 else 0,
            }
        )

        run_stats.append(stats)

    # Create run DataFrame
    exit_df = pd.DataFrame(run_stats).set_index("run")

    # Handle baseline comparison
    comparison = pd.DataFrame()
    if baseline_file and baseline_file in exit_df.index:
        # Get baseline stats
        baseline_stats = exit_df.loc[baseline_file]

        # Calculate differences for all other runs
        other_runs = exit_df.index != baseline_file
        comparison = pd.DataFrame({col: exit_df.loc[other_runs, col] - baseline_stats[col] for col in exit_df.columns})

    return exit_df, comparison


def load_log_files(
    logs_files: Union[str, List[str]],
    baseline_file: Optional[str] = None,
    repo_id: Optional[str] = None,
    no_cache: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load log files from local paths or HuggingFace.

    Args:
        logs_files: Path(s) to the JSONL logs file(s)
        baseline_file: Optional path to baseline JSONL file to compare against
        repo_id: Optional Hugging Face repo ID to download from
        no_cache: Whether to bypass cache when downloading

    Returns:
        Dict mapping file paths to their loaded contents
    """
    if isinstance(logs_files, str):
        logs_files = [logs_files]

    # Add baseline to logs_files if provided
    if baseline_file:
        logs_files = [*logs_files, baseline_file]

    results = {}
    for logs_file in logs_files:
        # Get file path (downloading if needed)
        try:
            file_path = get_file_path(logs_file, caller_name="view_logs", repo_id=repo_id, no_cache=no_cache)
        except:  # noqa: E722
            file_path = get_file_path(
                logs_file,
                caller_name="view_logs",
                repo_id="JetBrains-Research/EnvBench-trajectories",
                no_cache=no_cache,
            )

        results[logs_file] = load_jsonl(file_path)

    return results


def calculate_pass_rate(
    logs_files: Union[str, List[str]],
    baseline_file: Optional[str] = None,
    repo_id: Optional[str] = None,
    no_cache: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pass rates for multiple log files.

    Args:
        logs_files: Path(s) to the JSONL logs file(s)
        baseline_file: Optional path to baseline JSONL file to compare against
        repo_id: Optional Hugging Face repo ID to download from
        no_cache: Whether to bypass cache when downloading

    Returns:
        Tuple of DataFrames:
        - Per-run pass rates including baseline (index=run)
        - Comparison with baseline (index=run minus baseline, columns=pass_df columns)
    """
    # Load all files
    all_results = load_log_files(logs_files, baseline_file, repo_id, no_cache)

    # Calculate pass rates for each run
    run_stats = []
    for logs_file, results in all_results.items():
        total = len(results)
        if total == 0:
            run_stats.append(
                {
                    "run": logs_file,
                    "total": 0,
                    "success_count": 0,
                    "success_rate": 0.0,
                    "clean_count": 0,
                    "clean_rate": 0.0,
                }
            )
            continue

        # Count successes (exit_code = 0)
        success_count = sum(1 for r in results if r["exit_code"] == 0)

        # Count clean passes (exit_code = 0 and issues_count = 0)
        clean_count = sum(1 for r in results if r["exit_code"] == 0 and r.get("issues_count", 0) == 0)

        run_stats.append(
            {
                "run": logs_file,
                "total": total,
                "success_count": success_count,
                "success_rate": round(100 * success_count / total, 2),
                "clean_count": clean_count,
                "clean_rate": round(100 * clean_count / total, 2),
            }
        )

    # Create pass rate DataFrame
    pass_df = pd.DataFrame(run_stats).set_index("run")

    # Handle baseline comparison
    comparison = pd.DataFrame()
    if baseline_file and baseline_file in pass_df.index:
        # Get baseline stats
        baseline_stats = pass_df.loc[baseline_file]

        # Calculate differences for all other runs
        other_runs = pass_df.index != baseline_file
        comparison = pd.DataFrame({col: pass_df.loc[other_runs, col] - baseline_stats[col] for col in pass_df.columns})

    return pass_df, comparison
