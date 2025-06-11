from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph.graph import CompiledGraph

from ...async_bash_executor import CommandExecutionResult
from ...context_providers.build_instructions import EnvSetupInstructionProvider
from ...toolkits.base import BaseEnvSetupToolkit
from ...toolkits.shellcheck import ShellcheckToolkit
from ...utils import message_to_info
from ..base import BaseEnvSetupAgent
from .graph import create_shellcheck_workflow
from .state_schema import ShellcheckState, ShellcheckTrajectoryEntry, ShellcheckUpdate


class ShellcheckAgent(BaseEnvSetupAgent[ShellcheckState, ShellcheckUpdate, ShellcheckTrajectoryEntry]):
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
        return self._max_iterations

    @property
    def commands_history(self) -> List[CommandExecutionResult]:
        return self.toolkit.commands_history

    def get_agent(self) -> CompiledGraph:
        assert isinstance(self.toolkit, ShellcheckToolkit)
        return create_shellcheck_workflow(
            model=self.model,
            run_shellcheck_func=self.toolkit._get_readable_shellcheck_outputs,
            max_iterations=self.max_iterations,
        )

    def construct_initial_state(self, repository: str, revision: str, *args, **kwargs) -> ShellcheckState:
        return {}

    @staticmethod
    def process_update_for_trajectory(update: ShellcheckUpdate, *args, **kwargs) -> ShellcheckTrajectoryEntry:
        if "model" in update:
            node = "model"
            messages = update["model"].get("messages", [])
        elif "run_shellcheck" in update:
            node = "run_shellcheck"
            messages = update["run_shellcheck"].get("messages", [])
        else:
            node = (set(update.keys()) - {"timestamp"})[0]
            messages = []
        return {
            "timestamp": update["timestamp"],
            "node": node,
            "messages": [message_to_info(message) for message in messages],
        }
