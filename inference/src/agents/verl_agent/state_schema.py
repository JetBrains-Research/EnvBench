from typing import List, TypedDict

from inference.src.async_bash_executor import CommandExecutionResult
from inference.src.utils.messages_info import MessageInfo


class VerlAgentUpdate(TypedDict, total=False):
    # Generic update structure that can work with any graph state
    timestamp: str
    # Additional fields will be dynamically added based on the graph being used


class VerlAgentTrajectoryEntry(TypedDict, total=False):
    node: str
    messages: List[MessageInfo]
    commands: List[CommandExecutionResult]
    timestamp: str
