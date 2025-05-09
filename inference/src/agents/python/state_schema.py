from typing import Annotated, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep, RemainingSteps

from inference.src.async_bash_executor import CommandExecutionResult
from inference.src.utils.messages_info import MessageInfo


class EnvSetupPythonState(TypedDict, total=False):
    build_instructions: Optional[str]
    # fields from langgraph.prebuilt.chat_agent_executor.AgentState
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps


class EnvSetupPythonUpdate(TypedDict):
    agent: EnvSetupPythonState
    tools: EnvSetupPythonState
    timestamp: str


class EnvSetupPythonTrajectoryEntry(TypedDict, total=False):
    node: str
    messages: List[MessageInfo]
    commands: List[CommandExecutionResult]
    timestamp: str
