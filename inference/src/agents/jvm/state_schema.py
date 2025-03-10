from typing import Annotated, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep, RemainingSteps

from src.async_bash_executor import CommandExecutionResult
from src.utils.messages_info import MessageInfo


class EnvSetupJVMState(TypedDict, total=False):
    build_instructions: Optional[str]
    # fields from langgraph.prebuilt.chat_agent_executor.AgentState
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps


class EnvSetupJVMUpdate(TypedDict):
    agent: EnvSetupJVMState
    tools: EnvSetupJVMState
    timestamp: str


class EnvSetupJVMTrajectoryEntry(TypedDict, total=False):
    node: str
    messages: List[MessageInfo]
    commands: List[CommandExecutionResult]
    timestamp: str
