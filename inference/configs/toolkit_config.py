from enum import Enum
from typing import Dict, Optional

from src.async_bash_executor import AsyncBashExecutor
from src.toolkits import BashTerminalToolkit, JVMBashTerminalToolkit, PythonBashTerminalToolkit
from src.toolkits.base import BaseEnvSetupToolkit
from src.toolkits.installamatic import InstallamaticToolkit


class EnvSetupToolkit(Enum):
    bash = "bash"
    bash_jvm = "bash_jvm"
    bash_python = "bash_python"
    installamatic = "installamatic"

    async def instantiate(
        self,
        repository: str,
        revision: str,
        image: str,
        error_message: Optional[str],
        env_vars: Dict[str, str],
        repository_workdir: bool,
        container_start_timeout: int,
        bash_timeout: Optional[int],
        max_num_chars_bash_output: Optional[int],
        hf_name: str,
        output_dir: str,
        language: str,
        clear_repo: bool,
    ) -> BaseEnvSetupToolkit:
        bash_executor = await AsyncBashExecutor.create(
            repository=repository,
            revision=revision,
            image=image,
            error_message=error_message,
            env_vars=env_vars,
            repository_workdir=repository_workdir,
            container_start_timeout=container_start_timeout,
            bash_timeout=bash_timeout,
            max_num_chars_bash_output=max_num_chars_bash_output,
            hf_name=hf_name,
            output_dir=output_dir,
            language=language,
            clear_repo=clear_repo,
        )

        if self == EnvSetupToolkit.bash:
            return await BashTerminalToolkit.create(bash_executor=bash_executor)

        if self == EnvSetupToolkit.bash_jvm:
            return await JVMBashTerminalToolkit.create(bash_executor=bash_executor)

        if self == EnvSetupToolkit.bash_python:
            return await PythonBashTerminalToolkit.create(bash_executor=bash_executor)

        if self == EnvSetupToolkit.installamatic:
            return await InstallamaticToolkit.create(bash_executor=bash_executor)

        raise ValueError("Unknown configuration.")
