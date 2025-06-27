from typing import Callable, Generic, List, Optional, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from ...async_bash_executor import CommandExecutionResult
from ...utils import message_to_info
from ..base import BaseEnvSetupAgent
from .state_schema import VerlAgentTrajectoryEntry, VerlAgentUpdate

# Generic type for state
StateType = TypeVar("StateType")


def extract_bash_script(text: str) -> Optional[str]:
    """Extract bash script from AI message content."""
    import re

    # Look for bash code blocks
    bash_match = re.search(r"```bash(.*?)```", text, re.DOTALL)
    if bash_match:
        return bash_match.group(1).strip()
    return None


class VerlAgent(BaseEnvSetupAgent[StateType, VerlAgentUpdate, VerlAgentTrajectoryEntry], Generic[StateType]):
    def __init__(
        self,
        model: BaseChatModel,
        graph_partial: Callable[[BaseChatModel], CompiledStateGraph],
        max_iterations: Optional[int] = None,
    ):
        self.model = model
        self.graph_partial = graph_partial
        self._max_iterations = max_iterations
        self._commands_history: List[CommandExecutionResult] = []

    @property
    def max_iterations(self) -> Optional[int]:
        # This is a generic property that can be overridden by specific implementations
        return self._max_iterations

    @property
    def commands_history(self) -> List[CommandExecutionResult]:
        return self._commands_history

    def get_agent(self) -> CompiledStateGraph:
        return self.graph_partial(model=self.model)  # noqa

    def construct_initial_state(self, repository: str, revision: str, *args, **kwargs) -> StateType:
        # Generic initial state construction
        # This can be overridden by specific implementations
        return {"messages": [], "tools_kwargs": {"repository": repository, "revision": revision}}

    def process_update_for_trajectory(self, update: VerlAgentUpdate, *args, **kwargs) -> VerlAgentTrajectoryEntry:
        node = "unknown"
        messages = []
        last_ai_message = None

        # Process any node in the update
        for key, value in update.items():
            if key != "timestamp" and isinstance(value, dict):
                node = key
                messages = value.get("messages", [])
                # Track the last AI message
                for message in messages:
                    if isinstance(message, AIMessage):
                        last_ai_message = message
                break

        # If this is the final update and we have a last AI message, extract bash script
        if last_ai_message:
            bash_script = extract_bash_script(last_ai_message.content)
            if bash_script:
                command = CommandExecutionResult(command=bash_script, exit_code=0)
                self._commands_history.append(command)

        return {
            "timestamp": update.get("timestamp", ""),
            "node": node,
            "messages": [message_to_info(message) for message in messages],
            "commands": self._commands_history,
        }
