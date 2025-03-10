from typing import List

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field

from ..utils.modify_commands import add_flag_to_command
from .base import BaseEnvSetupToolkit


class BashTerminalToolkit(BaseEnvSetupToolkit):
    async def execute_bash_command(
        self,
        command: str = Field(
            description="A bash command with its arguments to be executed. It can modify the environment and files."
        ),
        reason: str = Field(
            description="A reason why you are calling the tool. For example, 'to change required gradle version' or 'to specify the java sdk'."
        ),
    ):
        """
        Executes a given bash command inside a Docker container.
        """
        # agent commands modified not to get stuck
        if "pyenv install" in command:
            command = add_flag_to_command(command, target_flag="f", target_flag_long="force")

        return (await self._execute_bash_command(command))[0]

    def get_tools(self, *args, **kwargs) -> List[BaseTool]:
        return [StructuredTool.from_function(coroutine=self.execute_bash_command)]
