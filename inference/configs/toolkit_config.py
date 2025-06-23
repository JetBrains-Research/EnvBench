from enum import Enum
from typing import Dict, Optional

from inference.src.async_bash_executor import AsyncBashExecutor
from inference.src.toolkits import BashTerminalToolkit, JVMBashTerminalToolkit, PythonBashTerminalToolkit
from inference.src.toolkits.base import BaseEnvSetupToolkit
from inference.src.toolkits.bash_terminal_readonly import BashTerminalReadOnlyToolkit
from inference.src.toolkits.installamatic import InstallamaticToolkit
from inference.src.toolkits.shellcheck import ShellcheckToolkit


class EnvSetupToolkit(Enum):
    bash = "bash"
    bash_jvm = "bash_jvm"
    bash_python = "bash_python"
    installamatic = "installamatic"
    shellcheck = "shellcheck"
    readonly = "readonly"

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
        # Determine if we need read-only mode
        read_only = self == EnvSetupToolkit.readonly

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
            read_only=read_only,
        )

        if self == EnvSetupToolkit.bash:
            return await BashTerminalToolkit.create(bash_executor=bash_executor)

        if self == EnvSetupToolkit.bash_jvm:
            return await JVMBashTerminalToolkit.create(bash_executor=bash_executor)

        if self == EnvSetupToolkit.bash_python:
            return await PythonBashTerminalToolkit.create(bash_executor=bash_executor)

        if self == EnvSetupToolkit.readonly:
            return await BashTerminalReadOnlyToolkit.create(bash_executor=bash_executor)

        if self == EnvSetupToolkit.installamatic:
            return await InstallamaticToolkit.create(bash_executor=bash_executor)

        if self == EnvSetupToolkit.shellcheck:
            return await ShellcheckToolkit.create(bash_executor=bash_executor)

        raise ValueError("Unknown configuration.")
