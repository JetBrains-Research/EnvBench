from typing import Optional

from pydantic import BaseModel

from .agent_config import EnvSetupAgentConfig
from .data_source_config import DataSourceConfig
from .docker_config import DockerConfig
from .hf_config import HFConfig


class EnvSetupRunnerConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    agent: EnvSetupAgentConfig
    """Configuration for the environment setup agent. Refer to EnvSetupAgentConfig for details."""
    data_source: DataSourceConfig
    """Configuration for data source, i.e., where data about repositories to run on comes from. 
      Refer to DataSourceConfig for details.
      Data should contain 'repository' and 'revision' keys."""
    docker: DockerConfig
    """Configuration for the Docker image the terminal runs in. Refer to DockerConfig for details."""
    hf: HFConfig
    """Configuration for uploading trajectories to HuggingFace. Refer to HFConfig for details."""
    langsmith_project: Optional[str] = None
    """Set to string with the name of the project to log to to enable LangSmith logging."""
    log_trajectory: bool
    """Set to True to save all messages from current interaction in jsonlines format."""
    logging_dir: str
    """Directory where trajectories are stored."""
    max_concurrent: Optional[int]
    """Limits the number of concurrently running coroutines if set."""
    rewrite_trajectories: bool
    """Set to True to process all data points regardless of whether they are already present in `logging_dir`,
      False to exclude those for which trajectories are already available."""
    global_timeout: Optional[int] = None
