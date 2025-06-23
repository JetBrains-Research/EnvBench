from operator import add
from typing import Annotated, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage

from inference.src.async_bash_executor import CommandExecutionResult
from inference.src.utils.messages_info import MessageInfo


class ShellcheckState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add]
    turn: Annotated[int, "Current turn number"]
    max_turns: Annotated[int, "Maximum number of turns allowed"]
    should_continue: bool | None
    script: str | None


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
