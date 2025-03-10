from typing import List, Optional

from datasets import get_dataset_config_names, load_dataset  # type: ignore[import-untyped, import-not-found]

from .base import BaseDataSource


class HFDataSource(BaseDataSource):
    """Class to iterate over a dataset from HuggingFace Hub."""

    def __init__(
        self,
        hub_name: str,
        configs: Optional[List[str]] = None,
        split: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self._hub_name = hub_name
        self._cache_dir = cache_dir

        if configs:
            self._configs = configs
        else:
            self._configs = get_dataset_config_names(self._hub_name)
        self._split = split

    def __iter__(self):
        for config in self._configs:
            dataset = load_dataset(self._hub_name, config, split=self._split, cache_dir=self._cache_dir)
            yield from dataset
