from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from langgraph.graph.graph import CompiledGraph

from src.async_bash_executor import CommandExecutionResult

StateUpdate = TypeVar("StateUpdate")
GraphState = TypeVar("GraphState")
TrajectoryEntry = TypeVar("TrajectoryEntry")


class BaseEnvSetupAgent(ABC, Generic[GraphState, StateUpdate, TrajectoryEntry]):
    @property
    def max_iterations(self) -> Optional[int]:
        return None

    @property
    def configurable_config(self) -> Dict[str, Any]:
        return {}

    @property
    @abstractmethod
    def commands_history(self) -> List[CommandExecutionResult]: ...

    @abstractmethod
    def get_agent(self) -> CompiledGraph: ...

    @abstractmethod
    def construct_initial_state(self, repository: str, revision: str, *args, **kwargs) -> GraphState: ...

    @staticmethod
    @abstractmethod
    def process_update_for_trajectory(update: StateUpdate, *args, **kwargs) -> TrajectoryEntry: ...
