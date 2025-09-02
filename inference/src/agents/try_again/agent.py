from typing import Callable, Generic, List, Optional, TypeVar, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, convert_to_messages
from langgraph.graph.state import CompiledStateGraph

from ...async_bash_executor import CommandExecutionResult
from ...utils import message_to_info
from ..base import BaseEnvSetupAgent
from .state_schema import TryAgainAgentTrajectoryEntry, TryAgainAgentUpdate
from ...toolkits.base import BaseEnvSetupToolkit
from envbench_graphs.try_again_paper.state_schema import TryAgainState
from envbench_graphs.try_again_paper.graph import create_try_again_workflow
from ...toolkits.bash_terminal import BashTerminalToolkit


def extract_bash_script(text: str) -> Optional[str]:
    """Extract bash script from AI message content."""
    import re
    
    bash_match = re.search(r"```bash(.*?)```", text, re.DOTALL)
    if bash_match:
        return bash_match.group(1).strip()
    return None


class TryAgainAgent(BaseEnvSetupAgent[TryAgainState, TryAgainAgentUpdate, TryAgainAgentTrajectoryEntry]):
    def __init__(
        self,
        model: BaseChatModel,
        toolkit: BaseEnvSetupToolkit,
        static_feedback: Optional[str] = None,
        reward_score_for_pass: float = 1.0,
        max_iterations: Optional[int] = None,
    ):
        self.model = model
        self.toolkit = toolkit
        self.static_feedback = static_feedback
        self.reward_score_for_pass = reward_score_for_pass
        self._max_iterations = max_iterations
        self._commands_history: List[CommandExecutionResult] = []

    @property
    def max_iterations(self) -> Optional[int]:
        # init state + each iteration is: model, run reward
        return 1 + 2 * self._max_iterations

    @property
    def commands_history(self) -> List[CommandExecutionResult]:
        return self._commands_history
    
    async def run_reward(self, message: AIMessage) -> Tuple[float, str]:
        # todo: make customizable??
        bash_script = extract_bash_script(message.content)

        result, exit_code = await self.toolkit._execute_bash_command(bash_script)

        if exit_code == 0:
            return 1.0, "Perfect"

        # if command failed, and we will proceed, let's do it with a clean session
        await self.toolkit.restart_container()

        if self.static_feedback:
            return 0.0, self.static_feedback
        
        return 0.0, result

    def get_agent(self) -> CompiledStateGraph:
        return create_try_again_workflow(model=self.model,
                                         run_reward_func=self.run_reward,
                                         reward_score_for_pass=self.reward_score_for_pass, 
                                         max_iterations=self._max_iterations)

    def construct_initial_state(self, repository: str, revision: str, *args, **kwargs) -> TryAgainState:
        if "messages" in kwargs:
            return {"messages": kwargs["messages"], "tools_kwargs": {"repository": repository, "revision": revision}}
        return {"tools_kwargs": {"repository": repository, "revision": revision}}

    def process_update_for_trajectory(self, update: TryAgainAgentUpdate, *args, **kwargs) -> TryAgainAgentTrajectoryEntry:
        node = "unknown"
        messages = []
        commands = []

        if "run_turn" in update:
            node = "run_turn"
            messages = update["run_turn"].get("messages", [])

            last_ai_message = None
            for message in messages[::-1]:
                if isinstance(message, AIMessage):
                    last_ai_message = message
                    break

            if last_ai_message:
                bash_script = extract_bash_script(last_ai_message.content)
                if bash_script:
                    command = CommandExecutionResult(command=bash_script, exit_code=0)
                    self._commands_history = [command]
        elif "initialize" in update:
            node = "initialize"
            messages = update["initialize"].get("messages", [])

        messages = convert_to_messages(messages)
        return {
            "timestamp": update["timestamp"],
            "node": node,
            "messages": [message_to_info(message) for message in messages],
            "commands": commands,
        }