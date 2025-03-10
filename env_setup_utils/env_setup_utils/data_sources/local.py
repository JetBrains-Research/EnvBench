import csv
import gzip
import json
from typing import Any, Dict, Iterator

import jsonlines

from .base import BaseDataSource


class LocalFileDataSource(BaseDataSource):
    """A wrapper to iterate over locally stored JSONLines or CSV file."""

    def __init__(self, path: str):
        self._path = path

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._path.endswith(".jsonl"):
            with jsonlines.open(self._path, "r") as jsonl_reader:
                yield from jsonl_reader
        elif self._path.endswith(".csv"):
            with open(self._path, "r") as f:
                csv_reader = csv.DictReader(f)
                yield from csv_reader
        elif self._path.endswith(".jsonl.gz"):
            with gzip.open(self._path, "rt") as f:
                for line in f:
                    yield json.loads(line)
        elif self._path.endswith(".csv.gz"):
            with gzip.open(self._path, "rt") as f:
                csv_reader = csv.DictReader(f)
                yield from csv_reader
        else:
            raise ValueError("Unsupported file format. Currently supported: .jsonl, .jsonl.gz, .csv, .csv.gz")
