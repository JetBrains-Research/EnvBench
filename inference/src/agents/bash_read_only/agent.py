from typing import List, Optional

from envbench_graphs.readonly import EnvSetupReadOnlyState, create_read_only_workflow
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph.graph import CompiledGraph

from ...async_bash_executor import CommandExecutionResult
from ...context_providers.build_instructions import EnvSetupInstructionProvider
from ...toolkits.base import BaseEnvSetupToolkit
from ...utils import message_to_info
from ..base import BaseEnvSetupAgent
from .state_schema import EnvSetupReadOnlyTrajectoryEntry, EnvSetupReadOnlyUpdate


class EnvSetupReadOnlyAgent(
    BaseEnvSetupAgent[EnvSetupReadOnlyState, EnvSetupReadOnlyUpdate, EnvSetupReadOnlyTrajectoryEntry]
):
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
        self._script: Optional[str] = None

    @property
    def max_iterations(self) -> Optional[int]:
        if self._max_iterations is None:
            return None
        return 2 * self._max_iterations + 3

    @property
    def commands_history(self) -> List[CommandExecutionResult]:
        if self._script is not None:
            return [CommandExecutionResult(command=self._script, exit_code=None)]
        return []

    def get_agent(self) -> CompiledGraph:
        execute_tools = self.toolkit.get_tools(stage="execute")
        submit_tools = self.toolkit.get_tools(stage="submit")
        return create_read_only_workflow(
            model=self.model,
            tools=execute_tools,
            submit_shell_script_tool=submit_tools[0],
            max_iterations=self._max_iterations,
        )

    def construct_initial_state(self, repository: str, revision: str, *args, **kwargs) -> EnvSetupReadOnlyState:
        return {"turn": 1}

    def process_update_for_trajectory(  # type: ignore[override]
        self, update: EnvSetupReadOnlyUpdate, *args, **kwargs
    ) -> EnvSetupReadOnlyTrajectoryEntry:
        node = "unknown"
        messages: List[BaseMessage] = []
        commands: List[CommandExecutionResult] = []

        if "init_state" in update:
            node = "init_state"
            messages = update["init_state"].get("messages", [])
        elif "model" in update:
            node = "model"
            messages = update["model"].get("messages", [])
        elif "tools" in update:
            node = "tools"
            messages = update["tools"].get("messages", [])
            # Extract commands from toolkit history if available
            if hasattr(self, "toolkit") and self.toolkit.commands_history:
                commands = self.toolkit.commands_history[-1:] if self.toolkit.commands_history else []
        elif "force_submit_shell_script_tool_call" in update:
            node = "force_submit_shell_script_tool_call"
            messages = update["force_submit_shell_script_tool_call"].get("messages", [])
        elif "submit_shell_script" in update:
            node = "submit_shell_script"
            messages = update["submit_shell_script"].get("messages", [])
            # Extract shell script if available in the state
            if "shell_script" in update["submit_shell_script"]:
                shell_script = update["submit_shell_script"]["shell_script"]
                self._script = shell_script
                if shell_script:
                    command = CommandExecutionResult(command=shell_script, exit_code=None)
                    commands = [command]

        return {
            "timestamp": update["timestamp"],
            "node": node,
            "messages": [message_to_info(message) for message in messages],
            "commands": commands,
        }
