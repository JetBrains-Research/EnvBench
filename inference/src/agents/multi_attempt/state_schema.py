from typing import List, TypedDict

from graphs.multi_attempt.state_schema import MultiAttemptState
from inference.src.async_bash_executor import CommandExecutionResult
from inference.src.utils.messages_info import MessageInfo


class MultiAttemptUpdate(TypedDict, total=False):
    initialize: MultiAttemptState
    model: MultiAttemptState
    next_turn: MultiAttemptState
    timestamp: str


class MultiAttemptTrajectoryEntry(TypedDict, total=False):
    node: str
    messages: List[MessageInfo]
    commands: List[CommandExecutionResult]
    timestamp: str
