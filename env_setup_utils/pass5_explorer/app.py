from datetime import datetime
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# HuggingFace API configuration
DATASET_NAME = "JetBrains-Research/envbench-rl-trajectories"
SPLIT_DATASET_NAME = "JetBrains-Research/envbench-zeroshot-rl"
hf_api = HfApi()
fs = HfFileSystem()

# Global file list - populated once at startup
_results_files = None
CACHE_FILE = Path(__file__).resolve().parent / "results_files_cache.json"

# Global split data - cached
_train_split_data = None
_test_split_data = None
SPLIT_CACHE_FILE = Path(__file__).resolve().parent / "split_data_cache.json"

# --- Add global for initial issues counts ---
INITIAL_ISSUES_FILE = Path(__file__).resolve().parent / "initial_issues_counts.jsonl"
_initial_issues_map = None


def get_initial_issues_map():
    global _initial_issues_map
    if _initial_issues_map is not None:
        return _initial_issues_map
    issues_map = {}
    try:
        with open(INITIAL_ISSUES_FILE, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    repo = data["repository"]
                    count = data["initial_issues_count"]
                    issues_map[repo] = count
    except Exception as e:
        logger.error(f"Failed to load initial issues: {e}")
    _initial_issues_map = issues_map
    return issues_map


# --- Only walk dataset on explicit recache, never on normal get/search ---
def walk_dataset_for_results(force_recache=False):
    """Walk the dataset directory looking for results.jsonl files, avoiding trajectories folders and skipping known folders with results.jsonl. Append new entries only. Only walk if force_recache=True."""
    global _results_files

    # If not forcing recache, always use cache if present
    if not force_recache and _results_files is not None:
        return _results_files
    if not force_recache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                _results_files = json.load(f)
            return _results_files
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            _results_files = []
            return _results_files

    # If force_recache, do the walk and update cache
    cached_files = []
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached_files = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            cached_files = []

    # Build a set of known parent folders (relative to dataset_path) that already have results.jsonl
    known_folders = set()
    known_files = set()
    for entry in cached_files:
        path = entry["path"]
        known_files.add(path)
        parent = os.path.dirname(path)
        known_folders.add(parent)

    # Try different path formats
    possible_paths = [f"datasets/{DATASET_NAME}", DATASET_NAME, f"datasets/{DATASET_NAME}/main", f"{DATASET_NAME}/main"]
    dataset_path = None
    for path in possible_paths:
        try:
            dataset_path = path
            break
        except Exception:
            continue
    if dataset_path is None:
        _results_files = cached_files
        return _results_files

    def walk_directory(path, rel_path=""):
        results = []
        try:
            # If this folder is already known to have results.jsonl, skip it
            if rel_path in known_folders:
                return []
            items = fs.ls(path, detail=True)
            found_results = False
            for item in items:
                item_path = item["name"]
                item_type = item["type"]
                clean_path = item_path.replace(f"{dataset_path}/", "")
                # Skip trajectories folders
                if item_type == "directory" and "trajectories" == item_path.split("/")[-1]:
                    continue
                if item_type == "file" and item_path.endswith("results.jsonl"):
                    if clean_path not in known_files:
                        from datetime import datetime
                        import re

                        mtime = ""
                        patterns = [
                            (r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", "%Y-%m-%d_%H-%M-%S"),
                            (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
                            (r"(\d{8})", "%Y%m%d"),
                        ]
                        for pattern, format_str in patterns:
                            date_match = re.search(pattern, clean_path)
                            if date_match:
                                try:
                                    date_str = date_match.group(1)
                                    dt = datetime.strptime(date_str, format_str)
                                    mtime = dt.isoformat()
                                    break
                                except Exception:
                                    continue
                        if not mtime:
                            mtime = "1900-01-01T00:00:00"
                        results.append({"path": clean_path, "size": item.get("size", 0), "last_modified": mtime})
                elif item_type == "directory":
                    # Only go into subfolders if this folder is not known to have results.jsonl
                    sub_rel_path = (
                        os.path.join(rel_path, os.path.basename(item_path)) if rel_path else os.path.basename(item_path)
                    )
                    # If this subfolder is known to have results.jsonl, skip recursion
                    if sub_rel_path in known_folders:
                        continue
                    sub_results = walk_directory(item_path, sub_rel_path)
                    results.extend(sub_results)
        except Exception:
            pass
        return results

    new_results = walk_directory(dataset_path, "")
    all_results = cached_files + new_results
    seen = set()
    deduped_results = []
    for entry in all_results:
        if entry["path"] not in seen:
            deduped_results.append(entry)
            seen.add(entry["path"])
    deduped_results.sort(key=lambda x: x.get("last_modified", "1900-01-01T00:00:00"), reverse=True)
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(deduped_results, f, indent=2)
    except Exception:
        pass
    _results_files = deduped_results
    return _results_files


# Update get_results_files and search_results_files to never walk unless forced


def get_results_files():
    return walk_dataset_for_results(force_recache=False)


def search_results_files(query=""):
    files = get_results_files()
    results_files = []
    for file_info in files:
        if query.lower() in file_info["path"].lower():
            results_files.append(file_info)
    return results_files


def load_split_data():
    """Load train and test split data from the envbench-zeroshot-rl dataset"""
    global _train_split_data, _test_split_data

    # Check if we have cached data
    if _train_split_data is not None and _test_split_data is not None:
        return _train_split_data, _test_split_data

    # Try to load from cache file first
    if os.path.exists(SPLIT_CACHE_FILE):
        try:
            logger.info(f"Loading split data from cache: {SPLIT_CACHE_FILE}")
            with open(SPLIT_CACHE_FILE, "r") as f:
                cache_data = json.load(f)
                _train_split_data = cache_data.get("train", [])
                _test_split_data = cache_data.get("test", [])
            logger.info(f"Loaded {len(_train_split_data)} train and {len(_test_split_data)} test repos from cache")
            return _train_split_data, _test_split_data
        except Exception as e:
            logger.warning(f"Failed to load split cache: {e}")

    # If no cache, load from HuggingFace
    logger.info("No split cache found, loading from HuggingFace...")

    try:
        from datasets import load_dataset

        # Load train split
        logger.info("Loading train split...")
        train_dataset = load_dataset(SPLIT_DATASET_NAME, split="train")
        _train_split_data = [{"repository": item["repository"], "revision": item["revision"]} for item in train_dataset]

        # Load test split
        logger.info("Loading test split...")
        test_dataset = load_dataset(SPLIT_DATASET_NAME, split="test")
        _test_split_data = [{"repository": item["repository"], "revision": item["revision"]} for item in test_dataset]

        logger.info(f"Loaded {len(_train_split_data)} train and {len(_test_split_data)} test repos")

        # Save to cache
        try:
            cache_data = {"train": _train_split_data, "test": _test_split_data}
            with open(SPLIT_CACHE_FILE, "w") as f:
                json.dump(cache_data, f, indent=2)
            logger.info("Split data saved to cache")
        except Exception as e:
            logger.error(f"Failed to save split cache: {e}")

        return _train_split_data, _test_split_data

    except Exception as e:
        logger.error(f"Error loading split data: {e}")
        _train_split_data = []
        _test_split_data = []
        return _train_split_data, _test_split_data


def filter_results_by_split(results: List[Dict], split_data: List[Dict]) -> List[Dict]:
    """Filter results to only include repos that are in the given split"""
    if not split_data:
        return []

    # Create a set of (repository, revision) tuples for fast lookup
    split_repos = set()
    for item in split_data:
        repo = item["repository"]
        revision = item["revision"]
        split_repos.add((repo, revision))

    filtered_results = []
    for result in results:
        repo_name = result.get("repo_name", "")
        commit_sha = result.get("commit_sha", "")

        # Check if this result matches any repo in the split
        if (repo_name, commit_sha) in split_repos:
            filtered_results.append(result)

    logger.debug(f"Filtered {len(results)} results to {len(filtered_results)} for split")
    return filtered_results


def download_results_file(file_path: str) -> List[Dict]:
    """Download and parse a results.jsonl file using HfApi"""
    try:
        # Download the file using hf_hub_download which handles caching
        local_path = hf_hub_download(repo_id=DATASET_NAME, repo_type="dataset", filename=file_path)

        results = []
        with open(local_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        results.append(data)
                    except json.JSONDecodeError:
                        continue

        return results
    except Exception as e:
        print(f"Error downloading file {file_path}: {e}")
        return []


def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyze results and return statistics"""
    passed = 0
    has_issues = 0
    failed = 0
    total = len(results)
    total_issues_for_passed = 0  # Sum of issues_count for exit_code=0
    passed_with_exit_0 = 0  # Count of repos with exit_code=0 (passed + has_issues)

    logger.info(f"Analyzing {total} results")

    for i, result in enumerate(results):
        # Debug first few results to understand structure
        if i < 3:
            logger.info(f"Result {i}: {result}")

        # Try different possible structures
        exit_code = None
        issues_count = None

        # Structure 1: result.results.exit_code
        if "results" in result and isinstance(result["results"], dict):
            exit_code = result["results"].get("exit_code")
            issues_count = result["results"].get("issues_count", 0)
        # Structure 2: result.exit_code (direct)
        elif "exit_code" in result:
            exit_code = result["exit_code"]
            issues_count = result.get("issues_count", 0)
        # Structure 3: result.results might be a string or different format
        else:
            logger.warning(f"Unknown result structure: {result}")
            exit_code = 1  # Assume failed if we can't parse
            issues_count = 0

        # Default to failed if we can't find exit_code
        if exit_code is None:
            exit_code = 1
            issues_count = 0

        logger.debug(f"Repo: {result.get('repo_name', 'unknown')}, exit_code: {exit_code}, issues: {issues_count}")

        if exit_code == 0 and issues_count == 0:
            passed += 1
            passed_with_exit_0 += 1
            total_issues_for_passed += issues_count
        elif exit_code == 0:
            has_issues += 1
            passed_with_exit_0 += 1
            total_issues_for_passed += issues_count
        else:
            failed += 1

    # Calculate average errors for repos with exit_code=0
    avg_errs = total_issues_for_passed / passed_with_exit_0 if passed_with_exit_0 > 0 else 0

    logger.info(f"Analysis complete: {passed} passed, {has_issues} has issues, {failed} failed")
    logger.info(
        f"AvgErrs: {avg_errs:.2f} (total issues: {total_issues_for_passed}, exit_code=0 count: {passed_with_exit_0})"
    )

    return {
        "passed": passed,
        "has_issues": has_issues,
        "failed": failed,
        "total": total,
        "pass_rate": (passed / total * 100) if total > 0 else 0,
        "avg_errs": round(avg_errs, 2),
    }


def analyze_results_with_splits(results: List[Dict]) -> Dict[str, Any]:
    """Analyze results for both full dataset and train/test splits"""
    try:
        train_data, test_data = load_split_data()
    except Exception as e:
        logger.error(f"Failed to load split data: {e}")
        train_data, test_data = [], []

    # Full dataset analysis (existing functionality)
    full_analysis = analyze_results(results)

    # Train split analysis
    train_results = filter_results_by_split(results, train_data)
    train_analysis = (
        analyze_results(train_results)
        if train_results
        else {"passed": 0, "has_issues": 0, "failed": 0, "total": 0, "pass_rate": 0, "avg_errs": 0}
    )

    # Test split analysis
    test_results = filter_results_by_split(results, test_data)
    test_analysis = (
        analyze_results(test_results)
        if test_results
        else {"passed": 0, "has_issues": 0, "failed": 0, "total": 0, "pass_rate": 0, "avg_errs": 0}
    )

    logger.info(f"Split analysis - Train: {train_analysis['total']} repos, Test: {test_analysis['total']} repos")

    return {
        "full": full_analysis,
        "train": train_analysis,
        "test": test_analysis,
        "split_info": {
            "train_total": len(train_data),
            "test_total": len(test_data),
            "train_matched": train_analysis["total"],
            "test_matched": test_analysis["total"],
        },
    }


# --- Update calculate_cross_run_stats to add new metrics ---
def calculate_cross_run_stats(selected_files: List[str]) -> Dict[str, Any]:
    """Calculate cross-run statistics, including issue resolution rates."""
    all_repos = set()
    passed_repos = set()
    run_pass_counts = []
    run_has_issues_counts = []
    run_failed_counts = []
    run_avg_errs = []
    run_total_issues_solved_rate = []
    run_avg_issues_solved_rate = []

    initial_issues_map = get_initial_issues_map()

    for file_path in selected_files:
        results = download_results_file(file_path)
        run_passed = 0
        run_issues_count_sum = 0
        run_initial_issues_sum = 0
        run_issues_solved_rates = []

        run_analysis = analyze_results(results)
        run_avg_errs.append(run_analysis["avg_errs"])
        run_has_issues_counts.append(run_analysis["has_issues"])
        run_failed_counts.append(run_analysis["failed"])

        for result in results:
            repo_name = result.get("repo_name", "")
            if repo_name:
                all_repos.add(repo_name)
                exit_code = None
                issues_count = None
                if "results" in result and isinstance(result["results"], dict):
                    exit_code = result["results"].get("exit_code")
                    issues_count = result["results"].get("issues_count", 0)
                elif "exit_code" in result:
                    exit_code = result["exit_code"]
                    issues_count = result.get("issues_count", 0)
                else:
                    exit_code = 1
                    issues_count = 0
                if exit_code is None:
                    exit_code = 1
                    issues_count = 0
                if exit_code == 0 and issues_count == 0:
                    passed_repos.add(repo_name)
                    run_passed += 1
                # --- Issue solved rate logic ---
                initial_issues = initial_issues_map.get(repo_name)
                if initial_issues is not None and initial_issues > 0:
                    # If failed, use initial_issues_count as issues_count (assume no progress)
                    if exit_code != 0:
                        issues_count_for_metric = initial_issues
                    else:
                        issues_count_for_metric = issues_count
                    run_issues_count_sum += issues_count_for_metric
                    run_initial_issues_sum += initial_issues
                    run_issues_solved_rates.append(1 - (issues_count_for_metric / initial_issues))
        run_pass_counts.append(run_passed)
        # --- Compute rates for this run ---
        if run_initial_issues_sum > 0:
            total_issues_solved_rate = 1 - (run_issues_count_sum / run_initial_issues_sum)
        else:
            total_issues_solved_rate = 0
        avg_issues_solved_rate = (
            sum(run_issues_solved_rates) / len(run_issues_solved_rates) if run_issues_solved_rates else 0
        )
        run_total_issues_solved_rate.append(total_issues_solved_rate)
        run_avg_issues_solved_rate.append(avg_issues_solved_rate)

    k = len(selected_files)
    pass_at_k = len(passed_repos) / len(all_repos) * 100 if all_repos else 0
    pass_at_5 = None
    pass_at_5_repos = None
    all_samples_pass_at_5 = None
    if k > 5:
        # For pass@5, find repos that passed in at least one of the first 5 runs
        pass_5_repos = set()
        for i, file_path in enumerate(selected_files[:5]):
            results = download_results_file(file_path)
            for result in results:
                repo_name = result.get("repo_name", "")
                if repo_name:
                    exit_code = None
                    issues_count = None
                    if "results" in result and isinstance(result["results"], dict):
                        exit_code = result["results"].get("exit_code")
                        issues_count = result["results"].get("issues_count", 0)
                    elif "exit_code" in result:
                        exit_code = result["exit_code"]
                        issues_count = result.get("issues_count", 0)
                    else:
                        exit_code = 1
                        issues_count = 0
                    if exit_code is None:
                        exit_code = 1
                        issues_count = 0
                    if exit_code == 0 and issues_count == 0:
                        pass_5_repos.add(repo_name)
        pass_at_5_repos = len(pass_5_repos)
        pass_at_5 = pass_at_5_repos / len(all_repos) * 100 if all_repos else 0
        # --- New: mean pass@5 over all samples, computed efficiently ---
        from math import comb

        # For each repo, count number of successful runs
        repo_success_counts = {repo: 0 for repo in all_repos}
        for file_path in selected_files:
            results = download_results_file(file_path)
            for result in results:
                repo_name = result.get("repo_name", "")
                if repo_name in repo_success_counts:
                    exit_code = None
                    issues_count = None
                    if "results" in result and isinstance(result["results"], dict):
                        exit_code = result["results"].get("exit_code")
                        issues_count = result["results"].get("issues_count", 0)
                    elif "exit_code" in result:
                        exit_code = result["exit_code"]
                        issues_count = result.get("issues_count", 0)
                    else:
                        exit_code = 1
                        issues_count = 0
                    if exit_code is None:
                        exit_code = 1
                        issues_count = 0
                    if exit_code == 0 and issues_count == 0:
                        repo_success_counts[repo_name] += 1
        n = k
        mean_pass_at_5_probs = []
        for repo, m in repo_success_counts.items():
            if n < 5:
                prob = 1.0 if m > 0 else 0.0
            elif m == 0:
                prob = 0.0
            elif m == n:
                prob = 1.0
            else:
                try:
                    prob = 1 - (comb(n - m, 5) / comb(n, 5))
                except Exception:
                    prob = 0.0
            mean_pass_at_5_probs.append(prob)
        if mean_pass_at_5_probs:
            all_samples_pass_at_5 = sum(mean_pass_at_5_probs) / len(mean_pass_at_5_probs) * 100
    avg_passed_repos = sum(run_pass_counts) / len(run_pass_counts) if run_pass_counts else 0
    avg_pass_at_1 = avg_passed_repos / len(all_repos) * 100 if all_repos else 0
    mean_avg_errs = sum(run_avg_errs) / len(run_avg_errs) if run_avg_errs else 0
    mean_has_issues = sum(run_has_issues_counts) / len(run_has_issues_counts) if run_has_issues_counts else 0
    mean_failed = sum(run_failed_counts) / len(run_failed_counts) if run_failed_counts else 0
    import math

    def calculate_std(values, mean_val):
        if len(values) <= 1:
            return 0
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    std_passed = calculate_std(run_pass_counts, avg_passed_repos)
    std_has_issues = calculate_std(run_has_issues_counts, mean_has_issues)
    std_failed = calculate_std(run_failed_counts, mean_failed)
    std_avg_errs = calculate_std(run_avg_errs, mean_avg_errs)
    mean_total_issues_solved_rate = (
        sum(run_total_issues_solved_rate) / len(run_total_issues_solved_rate) if run_total_issues_solved_rate else 0
    )
    mean_avg_issues_solved_rate = (
        sum(run_avg_issues_solved_rate) / len(run_avg_issues_solved_rate) if run_avg_issues_solved_rate else 0
    )
    std_total_issues_solved_rate = calculate_std(run_total_issues_solved_rate, mean_total_issues_solved_rate)
    std_avg_issues_solved_rate = calculate_std(run_avg_issues_solved_rate, mean_avg_issues_solved_rate)
    mean_total_issues_solved_rate_pct = mean_total_issues_solved_rate * 100
    mean_avg_issues_solved_rate_pct = mean_avg_issues_solved_rate * 100
    std_total_issues_solved_rate_pct = std_total_issues_solved_rate * 100
    std_avg_issues_solved_rate_pct = std_avg_issues_solved_rate * 100
    result = {
        "total_repos": len(all_repos),
        "passed_repos": len(passed_repos),
        "pass_at_k": pass_at_k,
        "avg_pass_at_1": avg_pass_at_1,
        "avg_passed_repos": round(avg_passed_repos, 1),
        "avg_errs": round(mean_avg_errs, 2),
        "avg_has_issues": round(mean_has_issues, 1),
        "avg_failed": round(mean_failed, 1),
        "std_passed": round(std_passed, 1),
        "std_has_issues": round(std_has_issues, 1),
        "std_failed": round(std_failed, 1),
        "std_avg_errs": round(std_avg_errs, 2),
        "k": k,
        "total_issue_resolution_rate": round(mean_total_issues_solved_rate_pct, 2),
        "mean_issue_resolution_rate": round(mean_avg_issues_solved_rate_pct, 2),
        "std_total_issue_resolution_rate": round(std_total_issues_solved_rate_pct, 2),
        "std_mean_issue_resolution_rate": round(std_avg_issues_solved_rate_pct, 2),
    }
    if k > 5:
        result["pass_at_5"] = pass_at_5
        result["pass_at_5_repos"] = pass_at_5_repos
        result["all_samples_pass_at_5"] = all_samples_pass_at_5
    return result


def calculate_cross_run_stats_with_splits(selected_files: List[str]) -> Dict[str, Any]:
    """Calculate cross-run statistics for both full dataset and train/test splits"""
    # Calculate original cross-run stats (for backward compatibility)
    full_stats = calculate_cross_run_stats(selected_files)

    # Load split data
    try:
        train_data, test_data = load_split_data()
    except Exception as e:
        logger.error(f"Failed to load split data for cross-run analysis: {e}")
        return {"full": full_stats, "train": None, "test": None, "error": f"Failed to load split data: {str(e)}"}

    if not train_data and not test_data:
        return {"full": full_stats, "train": None, "test": None, "error": "No split data available"}

    # Calculate train split stats
    train_stats = calculate_split_cross_run_stats(selected_files, train_data, "train")

    # Calculate test split stats
    test_stats = calculate_split_cross_run_stats(selected_files, test_data, "test")

    return {"full": full_stats, "train": train_stats, "test": test_stats}


def calculate_split_cross_run_stats(
    selected_files: List[str], split_data: List[Dict], split_name: str
) -> Dict[str, Any]:
    """Calculate cross-run statistics for a specific split (train or test), including issue resolution rates"""
    # Create lookup set for this split
    split_repos = set()
    for item in split_data:
        repo = item["repository"]
        revision = item["revision"]
        split_repos.add((repo, revision))

    all_repos = set()
    passed_repos = set()
    run_pass_counts = []
    run_has_issues_counts = []
    run_failed_counts = []
    run_avg_errs = []
    run_total_issues_solved_rate = []
    run_avg_issues_solved_rate = []

    initial_issues_map = get_initial_issues_map()

    for file_path in selected_files:
        results = download_results_file(file_path)

        # Filter results for this split
        split_results = filter_results_by_split(results, split_data)

        if not split_results:
            # No matching results in this run for this split
            run_pass_counts.append(0)
            run_has_issues_counts.append(0)
            run_failed_counts.append(0)
            run_avg_errs.append(0)
            run_total_issues_solved_rate.append(0)
            run_avg_issues_solved_rate.append(0)
            continue

        # Calculate analysis for this run's split
        run_analysis = analyze_results(split_results)
        run_avg_errs.append(run_analysis["avg_errs"])
        run_has_issues_counts.append(run_analysis["has_issues"])
        run_failed_counts.append(run_analysis["failed"])

        run_passed = 0
        run_issues_count_sum = 0
        run_initial_issues_sum = 0
        run_issues_solved_rates = []

        for result in split_results:
            repo_name = result.get("repo_name", "")
            commit_sha = result.get("commit_sha", "")

            if repo_name and commit_sha:
                all_repos.add((repo_name, commit_sha))

                # Use same parsing logic as analyze_results
                exit_code = None
                issues_count = None

                if "results" in result and isinstance(result["results"], dict):
                    exit_code = result["results"].get("exit_code")
                    issues_count = result["results"].get("issues_count", 0)
                elif "exit_code" in result:
                    exit_code = result["exit_code"]
                    issues_count = result.get("issues_count", 0)
                else:
                    exit_code = 1
                    issues_count = 0

                if exit_code is None:
                    exit_code = 1
                    issues_count = 0

                if exit_code == 0 and issues_count == 0:
                    passed_repos.add((repo_name, commit_sha))
                    run_passed += 1

                # Issue resolution rate logic
                initial_issues = initial_issues_map.get(repo_name)
                if initial_issues is not None and initial_issues > 0:
                    # If failed, use initial_issues_count as issues_count (assume no progress)
                    if exit_code != 0:
                        issues_count_for_metric = initial_issues
                    else:
                        issues_count_for_metric = issues_count
                    run_issues_count_sum += issues_count_for_metric
                    run_initial_issues_sum += initial_issues
                    run_issues_solved_rates.append(1 - (issues_count_for_metric / initial_issues))

        run_pass_counts.append(run_passed)

        # Compute rates for this run
        if run_initial_issues_sum > 0:
            total_issues_solved_rate = 1 - (run_issues_count_sum / run_initial_issues_sum)
        else:
            total_issues_solved_rate = 0
        avg_issues_solved_rate = (
            sum(run_issues_solved_rates) / len(run_issues_solved_rates) if run_issues_solved_rates else 0
        )
        run_total_issues_solved_rate.append(total_issues_solved_rate)
        run_avg_issues_solved_rate.append(avg_issues_solved_rate)

    k = len(selected_files)
    pass_at_k = len(passed_repos) / len(all_repos) * 100 if all_repos else 0

    # Calculate pass@5 if k > 5
    pass_at_5 = None
    pass_at_5_repos = None
    all_samples_pass_at_5 = None
    if k > 5:
        # For pass@5, find repos that passed in at least one of the first 5 runs
        pass_5_repos = set()
        for i, file_path in enumerate(selected_files[:5]):
            results = download_results_file(file_path)
            split_results = filter_results_by_split(results, split_data)

            for result in split_results:
                repo_name = result.get("repo_name", "")
                commit_sha = result.get("commit_sha", "")

                if repo_name and commit_sha:
                    exit_code = None
                    issues_count = None

                    if "results" in result and isinstance(result["results"], dict):
                        exit_code = result["results"].get("exit_code")
                        issues_count = result["results"].get("issues_count", 0)
                    elif "exit_code" in result:
                        exit_code = result["exit_code"]
                        issues_count = result.get("issues_count", 0)
                    else:
                        exit_code = 1
                        issues_count = 0

                    if exit_code is None:
                        exit_code = 1
                        issues_count = 0

                    if exit_code == 0 and issues_count == 0:
                        pass_5_repos.add((repo_name, commit_sha))

        pass_at_5_repos = len(pass_5_repos)
        pass_at_5 = pass_at_5_repos / len(all_repos) * 100 if all_repos else 0

        # Mean pass@5 over all samples
        from math import comb

        repo_success_counts = {f"{repo}:{sha}": 0 for repo, sha in all_repos}
        for file_path in selected_files:
            results = download_results_file(file_path)
            split_results = filter_results_by_split(results, split_data)
            for result in split_results:
                repo_name = result.get("repo_name", "")
                commit_sha = result.get("commit_sha", "")
                repo_key = f"{repo_name}:{commit_sha}"
                if repo_key in repo_success_counts:
                    exit_code = None
                    issues_count = None
                    if "results" in result and isinstance(result["results"], dict):
                        exit_code = result["results"].get("exit_code")
                        issues_count = result["results"].get("issues_count", 0)
                    elif "exit_code" in result:
                        exit_code = result["exit_code"]
                        issues_count = result.get("issues_count", 0)
                    else:
                        exit_code = 1
                        issues_count = 0
                    if exit_code is None:
                        exit_code = 1
                        issues_count = 0
                    if exit_code == 0 and issues_count == 0:
                        repo_success_counts[repo_key] += 1

        n = k
        mean_pass_at_5_probs = []
        for repo_key, m in repo_success_counts.items():
            if n < 5:
                prob = 1.0 if m > 0 else 0.0
            elif m == 0:
                prob = 0.0
            elif m == n:
                prob = 1.0
            else:
                try:
                    prob = 1 - (comb(n - m, 5) / comb(n, 5))
                except Exception:
                    prob = 0.0
            mean_pass_at_5_probs.append(prob)
        if mean_pass_at_5_probs:
            all_samples_pass_at_5 = sum(mean_pass_at_5_probs) / len(mean_pass_at_5_probs) * 100

    # Calculate averages
    avg_passed_repos = sum(run_pass_counts) / len(run_pass_counts) if run_pass_counts else 0
    avg_pass_at_1 = avg_passed_repos / len(all_repos) * 100 if all_repos else 0

    mean_avg_errs = sum(run_avg_errs) / len(run_avg_errs) if run_avg_errs else 0
    mean_has_issues = sum(run_has_issues_counts) / len(run_has_issues_counts) if run_has_issues_counts else 0
    mean_failed = sum(run_failed_counts) / len(run_failed_counts) if run_failed_counts else 0

    # Calculate standard deviations
    import math

    def calculate_std(values, mean_val):
        if len(values) <= 1:
            return 0
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    std_passed = calculate_std(run_pass_counts, avg_passed_repos)
    std_has_issues = calculate_std(run_has_issues_counts, mean_has_issues)
    std_failed = calculate_std(run_failed_counts, mean_failed)
    std_avg_errs = calculate_std(run_avg_errs, mean_avg_errs)

    # Issue resolution rates
    mean_total_issues_solved_rate = (
        sum(run_total_issues_solved_rate) / len(run_total_issues_solved_rate) if run_total_issues_solved_rate else 0
    )
    mean_avg_issues_solved_rate = (
        sum(run_avg_issues_solved_rate) / len(run_avg_issues_solved_rate) if run_avg_issues_solved_rate else 0
    )
    std_total_issues_solved_rate = calculate_std(run_total_issues_solved_rate, mean_total_issues_solved_rate)
    std_avg_issues_solved_rate = calculate_std(run_avg_issues_solved_rate, mean_avg_issues_solved_rate)
    mean_total_issues_solved_rate_pct = mean_total_issues_solved_rate * 100
    mean_avg_issues_solved_rate_pct = mean_avg_issues_solved_rate * 100
    std_total_issues_solved_rate_pct = std_total_issues_solved_rate * 100
    std_avg_issues_solved_rate_pct = std_avg_issues_solved_rate * 100

    logger.info(f"{split_name.title()} cross-run stats: {len(all_repos)} total repos, {len(passed_repos)} passed repos")
    logger.info(f"{split_name.title()} Pass@k: {len(passed_repos)}/{len(all_repos)} = {pass_at_k:.1f}%")

    result = {
        "total_repos": len(all_repos),
        "passed_repos": len(passed_repos),
        "pass_at_k": pass_at_k,
        "avg_pass_at_1": avg_pass_at_1,
        "avg_passed_repos": round(avg_passed_repos, 1),
        "avg_errs": round(mean_avg_errs, 2),
        "avg_has_issues": round(mean_has_issues, 1),
        "avg_failed": round(mean_failed, 1),
        "std_passed": round(std_passed, 1),
        "std_has_issues": round(std_has_issues, 1),
        "std_failed": round(std_failed, 1),
        "std_avg_errs": round(std_avg_errs, 2),
        "k": k,
        "split_total": len(split_data),
        "total_issue_resolution_rate": round(mean_total_issues_solved_rate_pct, 2),
        "mean_issue_resolution_rate": round(mean_avg_issues_solved_rate_pct, 2),
        "std_total_issue_resolution_rate": round(std_total_issues_solved_rate_pct, 2),
        "std_mean_issue_resolution_rate": round(std_avg_issues_solved_rate_pct, 2),
    }

    if k > 5:
        result["pass_at_5"] = pass_at_5
        result["pass_at_5_repos"] = pass_at_5_repos
        result["all_samples_pass_at_5"] = all_samples_pass_at_5

    return result


@app.route("/")
def index():
    return render_template("index.html")


def get_cache_info():
    """Get cache age and status information"""
    if os.path.exists(CACHE_FILE):
        try:
            stat = os.stat(CACHE_FILE)
            cache_age = time.time() - stat.st_mtime
            cache_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
            return {"exists": True, "age_seconds": cache_age, "last_modified": cache_modified}
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {"exists": False}
    else:
        return {"exists": False}


@app.route("/api/search_files")
def api_search_files():
    query = request.args.get("query", "")
    logger.info(f"API request to search files with query: '{query}'")

    try:
        files = search_results_files(query)
        cache_info = get_cache_info()
        logger.info(f"Returning {len(files)} files to frontend")
        return jsonify({"files": files, "cache_info": cache_info, "total_count": len(get_results_files())})
    except Exception as e:
        logger.error(f"Error in search_files API: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear_cache", methods=["POST"])
def api_clear_cache():
    """Clear the file cache and split cache, then re-walk/reload datasets"""
    global _results_files, _train_split_data, _test_split_data
    _results_files = None
    _train_split_data = None
    _test_split_data = None

    # Remove cache files
    cache_files_removed = []

    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
            cache_files_removed.append(str(CACHE_FILE))
            logger.info(f"Removed cache file: {CACHE_FILE}")
        except Exception as e:
            logger.error(f"Failed to remove cache file: {e}")

    if os.path.exists(SPLIT_CACHE_FILE):
        try:
            os.remove(SPLIT_CACHE_FILE)
            cache_files_removed.append(str(SPLIT_CACHE_FILE))
            logger.info(f"Removed split cache file: {SPLIT_CACHE_FILE}")
        except Exception as e:
            logger.error(f"Failed to remove split cache file: {e}")

    logger.info("All caches cleared, re-loading data...")

    # Immediately re-walk/reload datasets to rebuild caches
    try:
        files = walk_dataset_for_results(force_recache=True)
        train_data, test_data = load_split_data()
        logger.info(f"Results cache rebuilt with {len(files)} files")
        logger.info(f"Split cache rebuilt with {len(train_data)} train and {len(test_data)} test repos")

        cache_info = get_cache_info()
        return jsonify(
            {
                "success": True,
                "message": f"Caches rebuilt successfully with {len(files)} results files and {len(train_data) + len(test_data)} split repos",
                "file_count": len(files),
                "split_count": {"train": len(train_data), "test": len(test_data)},
                "cache_files_removed": cache_files_removed,
                "cache_info": cache_info,
            }
        )
    except Exception as e:
        logger.error(f"Error rebuilding caches: {e}")
        return jsonify({"success": False, "error": f"Failed to rebuild caches: {str(e)}"}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json()
    selected_files = data.get("files", [])

    if not selected_files:
        return jsonify({"error": "No files selected"})

    # Analyze individual files with splits
    file_analyses = {}
    for file_path in selected_files:
        results = download_results_file(file_path)
        # Use the new function that includes split analysis
        analysis = analyze_results_with_splits(results)
        file_analyses[file_path] = analysis

    # Calculate cross-run statistics with splits
    cross_run_stats = calculate_cross_run_stats_with_splits(selected_files)

    return jsonify({"file_analyses": file_analyses, "cross_run_stats": cross_run_stats})


if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")
