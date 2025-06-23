from typing import List, TypedDict

from graphs.readonly.state_schema import EnvSetupReadOnlyState
from inference.src.async_bash_executor import CommandExecutionResult
from inference.src.utils.messages_info import MessageInfo


class EnvSetupReadOnlyUpdate(TypedDict, total=False):
    init_state: EnvSetupReadOnlyState
    model: EnvSetupReadOnlyState
    tools: EnvSetupReadOnlyState
    force_submit_shell_script_tool_call: EnvSetupReadOnlyState
    submit_shell_script: EnvSetupReadOnlyState
    timestamp: str


class EnvSetupReadOnlyTrajectoryEntry(TypedDict, total=False):
    node: str
    messages: List[MessageInfo]
    commands: List[CommandExecutionResult]
    timestamp: str
