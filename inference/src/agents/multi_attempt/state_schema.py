from operator import add
from typing import Annotated, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage

from inference.src.async_bash_executor import CommandExecutionResult
from inference.src.utils.messages_info import MessageInfo


class MultiAttemptState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    turn: int


class MultiAttemptUpdate(TypedDict):
    initialize: MultiAttemptState
    model: MultiAttemptState
    next_turn: MultiAttemptState
    timestamp: str


class MultiAttemptTrajectoryEntry(TypedDict, total=False):
    node: str
    messages: List[MessageInfo]
    commands: List[CommandExecutionResult]
    timestamp: str
