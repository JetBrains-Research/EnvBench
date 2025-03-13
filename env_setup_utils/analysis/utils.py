"""Utility functions for evaluation scripts."""

from pathlib import Path
import tempfile
from typing import Optional, cast

from huggingface_hub import HfFileSystem, hf_hub_download

from env_setup_utils.analysis.cache_manager import CacheManager

DEFAULT_REPO = "JetBrains-Research/EnvBench-trajectories"
DEFAULT_FILES = {"scripts_viewer": "scripts.jsonl", "view_logs": "results.jsonl"}

# Initialize cache manager as a module-level singleton
_cache_manager = CacheManager()


def get_file_path(
    file_path: Optional[str] = None,
    caller_name: str = "",
    repo_id: Optional[str] = None,
    no_cache: bool = False,
) -> str:
    """Get the path to a data file, downloading it from Hugging Face if necessary.

    Args:
        file_path: Path to the file. If None, uses the default filename in a temp directory.
        caller_name: Name of the calling script (e.g., 'scripts_viewer' or 'view_logs').
        repo_id: Hugging Face repository ID to download from. If None, uses DEFAULT_REPO.
        no_cache: If True, bypass the cache and force redownload.

    Returns:
        Path to the file as a string.

    Raises:
        ValueError: If the caller_name is not recognized.
        RuntimeError: If the file cannot be downloaded or accessed.
    """
    if not caller_name:
        raise ValueError("caller_name must be specified")

    if caller_name not in DEFAULT_FILES:
        raise ValueError(f"Unknown caller: {caller_name}. Must be one of {list(DEFAULT_FILES.keys())}")

    # If no path provided, use the default filename in a temp directory
    if file_path is None:
        temp_dir = tempfile.gettempdir()
        file_path = str(Path(temp_dir) / DEFAULT_FILES[caller_name])

    path = Path(cast(str, file_path))

    # If file exists and we're not bypassing cache, return its path
    if path.exists() and not no_cache:
        return str(path)

    # If path is a directory, append the default filename
    if path.is_dir() or path.suffix == "":
        path = path / DEFAULT_FILES[caller_name]

    try:
        # Check cache first
        repo = repo_id if repo_id is not None else DEFAULT_REPO
        cached_path = _cache_manager.get_cached_path(repo, str(path), no_cache)
        if cached_path:
            return cached_path

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Download using hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=repo,
            filename=str(path),
            repo_type="dataset",
            local_dir=str(path.parent),
            local_dir_use_symlinks=False,
            force_download=no_cache,
        )

        # Update cache with the new file
        _cache_manager.update_cache(repo, str(path), downloaded_path)

        return downloaded_path

    except Exception as e:
        raise RuntimeError(f"Failed to download file from {repo_id if repo_id is not None else DEFAULT_REPO}: {e}")


def get_dir_path(
    dir_path: str,
    repo_id: Optional[str] = None,
    no_cache: bool = False,
) -> str:
    """Get the path to a directory, downloading it from Hugging Face if necessary.

    The function maps local paths to Hugging Face repository paths. For example:
    'oss/python_singlerepo-2025-01-02/trajectories' will be downloaded from
    'JetBrains-Research/EnvBench-trajectories/oss/python_singlerepo-2025-01-02/trajectories'

    Args:
        dir_path: Path to the directory (e.g., 'oss/python_singlerepo-2025-01-02/trajectories').
        repo_id: Hugging Face repository ID to download from. If None, uses DEFAULT_REPO.
        no_cache: If True, bypass the cache and force redownload.

    Returns:
        Path to the directory as a string.

    Raises:
        RuntimeError: If the directory cannot be downloaded or accessed.
    """
    path = Path(dir_path)

    # If directory exists and we're not bypassing cache, return its path
    if path.exists() and any(path.iterdir()) and not no_cache:
        return str(path)

    try:
        # Create the directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Initialize HfFileSystem for listing files
        fs = HfFileSystem()
        repo = repo_id if repo_id is not None else DEFAULT_REPO

        # List all files in the directory
        pattern = f"datasets/{repo}/{dir_path}/*.jsonl"
        files = fs.glob(pattern)
        if not files:
            raise RuntimeError(f"No .jsonl files found matching {pattern}")

        # Download each file using hf_hub_download
        for hf_path in files:
            # Extract the relative path from the full HF path
            # Format: datasets/owner/repo/path/to/file.jsonl
            # We need: path/to/file.jsonl
            relative_path = "/".join(hf_path.split("/")[3:])
            target_file = path / Path(relative_path).name

            # Check cache first
            cached_path = _cache_manager.get_cached_path(repo, relative_path, no_cache)
            if cached_path:
                # Create symlink from cache to target if it doesn't exist
                if not target_file.exists():
                    target_file.symlink_to(cached_path)
                continue

            # Download the file to cache directory
            downloaded_path = hf_hub_download(
                repo_id=repo,
                filename=relative_path,
                repo_type="dataset",
                local_dir=str(_cache_manager.cache_dir),
                local_dir_use_symlinks=False,
                force_download=no_cache,
            )

            # Update cache with the new file
            _cache_manager.update_cache(repo, relative_path, downloaded_path)

            # Create symlink from cache to target
            if not target_file.exists():
                target_file.symlink_to(downloaded_path)

        if not any(path.iterdir()):
            raise RuntimeError(f"No files were downloaded from {dir_path}")
        else:
            print(f"Downloaded {len(list(path.iterdir()))} files from {dir_path}")

        return str(path)

    except Exception as e:
        raise RuntimeError(f"Failed to download directory from {repo_id if repo_id is not None else DEFAULT_REPO}: {e}")
