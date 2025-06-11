from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph.graph import CompiledGraph

from ...async_bash_executor import CommandExecutionResult
from ...context_providers.build_instructions import EnvSetupInstructionProvider
from ...toolkits.base import BaseEnvSetupToolkit
from ...utils import message_to_info
from ..base import BaseEnvSetupAgent
from .graph import create_multi_attempt_workflow
from .state_schema import MultiAttemptState, MultiAttemptTrajectoryEntry, MultiAttemptUpdate


class MultiAttemptAgent(BaseEnvSetupAgent[MultiAttemptState, MultiAttemptUpdate, MultiAttemptTrajectoryEntry]):
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
        return create_multi_attempt_workflow(
            model=self.model,
            max_iterations=self.max_iterations,
        )

    def construct_initial_state(self, repository: str, revision: str, *args, **kwargs) -> MultiAttemptState:
        return {"messages": [], "turn": 0}

    @staticmethod
    def process_update_for_trajectory(update: MultiAttemptUpdate, *args, **kwargs) -> MultiAttemptTrajectoryEntry:
        if "model" in update:
            node = "model"
            messages = update["model"].get("messages", [])
        elif "next_turn" in update:
            node = "next_turn"
            messages = update["next_turn"].get("messages", [])
        else:
            node = (set(update.keys()) - {"timestamp"})[0]
            messages = []
        return {
            "timestamp": update["timestamp"],
            "node": node,
            "messages": [message_to_info(message) for message in messages],
        }
