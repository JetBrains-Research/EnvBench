from typing import List

from env_setup_utils.data_sources import HFDataSource, LocalFileDataSource
from pydantic import BaseModel

from .instantiatable_config import InstantiatableConfig


class HFDataSourceConfig(InstantiatableConfig[HFDataSource]):
    hub_name: str
    configs: List[str]
    split: str


class LocalFileDataSourceConfig(InstantiatableConfig[LocalFileDataSource]):
    path: str


class DataSourceConfig(BaseModel):
    type: str
    hf: HFDataSourceConfig
    local: LocalFileDataSourceConfig
