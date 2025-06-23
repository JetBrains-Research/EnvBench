import re
from typing import List, Optional

from envbench_graphs.multi_attempt import MultiAttemptState, create_multi_attempt_workflow
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph.graph import CompiledGraph

from ...async_bash_executor import CommandExecutionResult
from ...context_providers.build_instructions import EnvSetupInstructionProvider
from ...toolkits.base import BaseEnvSetupToolkit
from ...utils import message_to_info
from ..base import BaseEnvSetupAgent
from .state_schema import MultiAttemptTrajectoryEntry, MultiAttemptUpdate


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
        self._max_iterations = max_iterations if max_iterations else 2
        self._resulting_command: Optional[CommandExecutionResult] = None

    @property
    def max_iterations(self) -> Optional[int]:
        # init state + each iteration is: model, next_turn
        return 1 + 2 * self._max_iterations

    @property
    def commands_history(self) -> List[CommandExecutionResult]:
        return [self._resulting_command]

    def get_agent(self) -> CompiledGraph:
        return create_multi_attempt_workflow(
            model=self.model,
            max_iterations=self._max_iterations,
        )

    def construct_initial_state(self, repository: str, revision: str, *args, **kwargs) -> MultiAttemptState:
        if "messages" in kwargs:
            return {"messages": kwargs["messages"]}
        return {}

    def process_update_for_trajectory(self, update: MultiAttemptUpdate, *args, **kwargs) -> MultiAttemptTrajectoryEntry:  # type: ignore[override]
        node = "unknown"
        messages: List[BaseMessage] = []
        commands: List[CommandExecutionResult] = []

        if "initialize" in update:
            node = "initialize"
            messages = update["initialize"].get("messages", [])
        elif "model" in update:
            node = "model"
            messages = update["model"].get("messages", [])

            matches = re.findall(r"```bash(.*?)```", messages[-1].content, re.DOTALL)
            if matches:
                script = matches[0].strip()
                command = CommandExecutionResult(command=script, exit_code=None)
                self._resulting_command = command
                commands = [command]

        elif "next_turn" in update:
            node = "next_turn"
            messages = update["next_turn"].get("messages", [])

        return {
            "timestamp": update["timestamp"],
            "node": node,
            "messages": [message_to_info(message) for message in messages],
            "commands": commands,
        }
