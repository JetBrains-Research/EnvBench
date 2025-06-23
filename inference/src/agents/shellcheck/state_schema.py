from typing import List, TypedDict

from graphs.shellcheck.state_schema import ShellcheckState
from inference.src.async_bash_executor import CommandExecutionResult
from inference.src.utils.messages_info import MessageInfo


class ShellcheckUpdate(TypedDict, total=False):
    initialize: ShellcheckState
    model: ShellcheckState
    process_output: ShellcheckState
    run_shellcheck: ShellcheckState
    timestamp: str


class ShellcheckTrajectoryEntry(TypedDict, total=False):
    node: str
    messages: List[MessageInfo]
    commands: List[CommandExecutionResult]
    timestamp: str
