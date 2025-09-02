from typing import List, TypedDict

from inference.src.async_bash_executor import CommandExecutionResult
from inference.src.utils.messages_info import MessageInfo

from envbench_graphs.try_again_paper.state_schema import TryAgainState

class TryAgainAgentUpdate(TypedDict, total=False):
    timestamp: str
    initialize: TryAgainState
    run_turn: TryAgainState

class TryAgainAgentTrajectoryEntry(TypedDict, total=False):
    node: str
    messages: List[MessageInfo]
    commands: List[CommandExecutionResult]
    timestamp: str
