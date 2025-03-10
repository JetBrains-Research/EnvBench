import os
from typing import Dict, Optional

from pydantic import BaseModel, validator


class DockerConfig(BaseModel):
    image: str
    """Image to build the container from."""
    command: Optional[str]
    """Command to run on the container start."""
    error_message: Optional[str]
    """Error message to return when the Bash command fails."""
    env_vars: Dict[str, str]
    """Environmental variables for the container."""
    repository_workdir: bool
    """Flag to determine if all the commands should be run in a repository root directory 
    or in default working directory (/)."""
    container_start_timeout: int
    """Time in which container is expected to start and execute the given command."""
    bash_timeout: Optional[int]
    """Timeout in seconds for bash commands execution."""
    max_num_chars_bash_output: Optional[int]
    """Maximum number of characters in bash command output; output of each command will be truncated if provided."""
    hf_name: str
    """HuggingFace handle for the dataset with repositories sources."""
    output_dir: str
    """Local path to directory where the cloned repositories will be stored."""
    language: str
    """Language for the current run; used to determine the path to repositories' sources inside HuggingFace dataset."""
    clear_repo: bool
    """Set to True to remove each downloaded repository after execution finishes, False to keep it."""

    @validator("env_vars", pre=True)
    def set_env_vars(cls, env_vars: Dict[str, Optional[str]]) -> Dict[str, str]:
        str_env_vars: Dict[str, str] = {key: value for key, value in env_vars.items() if isinstance(value, str)}
        assert set(str_env_vars.keys()) == set(
            env_vars.keys()
        ), f"Some variables have non-string values: {set(env_vars.keys()) - set(str_env_vars.keys())}"
        return str_env_vars

    @validator("output_dir", pre=True)
    def set_output_dir(cls, output_dir: str) -> str:
        if output_dir.startswith("~"):
            output_dir = f"/{os.path.expanduser('~')}/{output_dir[len('~/'):]}"
        return output_dir
