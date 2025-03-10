"""Cache manager for HuggingFace downloads."""

import json
import os
from pathlib import Path
from typing import Dict, Optional


class CacheManager:
    """Manages caching of HuggingFace downloads."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files. If None, uses ~/.hf_cache
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.hf_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "cache_map.json"
        self._cache_map: Dict[str, str] = self._load_cache_map()

    def _load_cache_map(self) -> Dict[str, str]:
        """Load the cache map from disk."""
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _save_cache_map(self) -> None:
        """Save the cache map to disk."""
        with open(self.cache_file, "w") as f:
            json.dump(self._cache_map, f, indent=2)

    def get_cached_path(self, repo_id: str, file_path: str, no_cache: bool = False) -> Optional[str]:
        """Get the cached path for a file if it exists.

        Args:
            repo_id: HuggingFace repository ID
            file_path: Path to the file within the repository
            no_cache: If True, bypass the cache and return None

        Returns:
            Path to the cached file if it exists and no_cache is False, None otherwise
        """
        if no_cache:
            return None

        cache_key = f"{repo_id}/{file_path}"
        cached_path = self._cache_map.get(cache_key)
        if cached_path and os.path.exists(cached_path):
            return cached_path
        return None

    def update_cache(self, repo_id: str, file_path: str, local_path: str) -> None:
        """Update the cache with a new file location.

        Args:
            repo_id: HuggingFace repository ID
            file_path: Path to the file within the repository
            local_path: Local path where the file is stored
        """
        cache_key = f"{repo_id}/{file_path}"
        self._cache_map[cache_key] = local_path
        self._save_cache_map()

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self._cache_map.clear()
        self._save_cache_map()
