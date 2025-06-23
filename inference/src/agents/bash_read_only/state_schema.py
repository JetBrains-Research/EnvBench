from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from inference.src.async_bash_executor import CommandExecutionResult
from inference.src.utils.messages_info import MessageInfo


class EnvSetupReadOnlyState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    turn: int
    shell_script: Optional[str]


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
