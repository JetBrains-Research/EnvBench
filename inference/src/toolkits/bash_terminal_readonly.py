from typing import List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field

from ..async_bash_executor import AsyncBashExecutor
from .base import BaseEnvSetupToolkit


class BashTerminalReadOnlyToolkit(BaseEnvSetupToolkit):
    async def execute_bash_command(
        self,
        command: str = Field(
            description="A bash command with its arguments to be executed. Only READ operations are allowed."
        ),
        reason: str = Field(
            description="A reason why you are calling the tool. For example, 'to check Python version' or 'to list directory contents'."
        ),
    ):
        """
        Executes a given bash command inside a Docker container in read-only mode.
        Only read operations are allowed - any write operations will fail.
        """
        return (await self._execute_bash_command(command))[0]

    async def submit_shell_script(
        self,
        script: str = Field(description="A resulting shell script for environment setup."),
    ):
        """
        Registers a shell script for environment setup for the current repository.
        This script can (and should!) contain write operations.
        """
        return "Done! The script is submitted."

    def get_tools(self, stage: Optional[str] = None, *args, **kwargs) -> List[BaseTool]:
        if stage is None:
            return [
                StructuredTool.from_function(coroutine=self.execute_bash_command),
                StructuredTool.from_function(coroutine=self.submit_shell_script),
            ]

        if stage == "execute":
            return [StructuredTool.from_function(coroutine=self.execute_bash_command)]

        if stage == "submit":
            return [StructuredTool.from_function(coroutine=self.submit_shell_script)]

        raise ValueError(f"Invalid stage: {stage}")

    @classmethod
    async def create(
        cls,
        bash_executor: AsyncBashExecutor,
    ):
        # Ensure the executor is in read-only mode
        if not bash_executor.read_only:
            raise ValueError("BashTerminalReadOnlyToolkit requires a read-only AsyncBashExecutor")

        tools_provider = cls(bash_executor=bash_executor)

        # run initial commands
        for command in tools_provider.initial_commands():
            result, err_code = await tools_provider._execute_bash_command(command)
            if err_code != 0:
                raise ValueError(f"Couldn't execute initial command {command}. Output: {result}")
        return tools_provider
