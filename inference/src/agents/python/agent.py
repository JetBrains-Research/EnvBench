from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

from ...async_bash_executor import CommandExecutionResult
from ...context_providers.build_instructions import EnvSetupInstructionProvider
from ...toolkits.base import BaseEnvSetupToolkit
from ...utils import message_to_info
from ..base import BaseEnvSetupAgent
from .prompts import get_env_setup_python_prompt
from .state_schema import EnvSetupPythonState, EnvSetupPythonTrajectoryEntry, EnvSetupPythonUpdate


class EnvSetupPythonAgent(BaseEnvSetupAgent[EnvSetupPythonState, EnvSetupPythonUpdate, EnvSetupPythonTrajectoryEntry]):
    def __init__(
        self,
        model: BaseChatModel,
        toolkit: BaseEnvSetupToolkit,
        instruction_provider: EnvSetupInstructionProvider,
        max_iterations: Optional[int] = None,
    ):
        self.toolkit = toolkit
        self.model = model
        self.instruction_provider = instruction_provider
        self._max_iterations = max_iterations

    @property
    def max_iterations(self) -> Optional[int]:
        if self._max_iterations is None:
            return None
        return 2 * self._max_iterations + 1

    @property
    def commands_history(self) -> List[CommandExecutionResult]:
        return self.toolkit.commands_history

    def get_agent(self) -> CompiledGraph:
        tools = self.toolkit.get_tools()
        return create_react_agent(
            model=self.model, tools=tools, state_schema=EnvSetupPythonState, state_modifier=get_env_setup_python_prompt
        )

    def construct_initial_state(self, repository: str, revision: str, *args, **kwargs) -> EnvSetupPythonState:
        return {"build_instructions": self.instruction_provider(repository=repository, revision=revision)}

    @staticmethod
    def process_update_for_trajectory(update: EnvSetupPythonUpdate, *args, **kwargs) -> EnvSetupPythonTrajectoryEntry:
        if "agent" in update:
            node = "agent"
            messages = update["agent"].get("messages", [])
        elif "tools" in update:
            node = "tools"
            messages = update["tools"].get("messages", [])
        else:
            raise RuntimeError(
                f"Expected the update to come either from 'agent' or 'tools' nodes, but got {set(update.keys()) - {'timestamp'}}."
            )
        return {
            "timestamp": update["timestamp"],
            "node": node,
            "messages": [message_to_info(message) for message in messages],
        }
