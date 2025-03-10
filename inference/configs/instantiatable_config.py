from typing import Generic, TypeVar

from hydra.utils import instantiate
from pydantic import BaseModel, Field

InstantiatedType = TypeVar("InstantiatedType")


class InstantiatableConfig(BaseModel, Generic[InstantiatedType]):
    target: str = Field(..., alias="_target_")

    def instantiate(self, **kwargs) -> InstantiatedType:
        return instantiate(self.dict(by_alias=True), **kwargs, _convert_="partial")
