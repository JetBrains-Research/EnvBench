from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator


class BaseDataSource(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError()
