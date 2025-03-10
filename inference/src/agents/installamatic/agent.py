from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph

from ...async_bash_executor import CommandExecutionResult
from ...toolkits.base import BaseEnvSetupToolkit
from ...utils import message_to_info
from ..base import BaseEnvSetupAgent
from .build_graph import InstallamaticBuildGraph
from .search_graph import InstallamaticSearchGraph
from .state_schema import (
    InstallamaticConfigurable,
    InstallamaticState,
    InstallamaticTrajectoryEntry,
    InstallamaticUpdate,
)


class InstallamaticAgent(BaseEnvSetupAgent[InstallamaticState, InstallamaticUpdate, InstallamaticTrajectoryEntry]):
    def __init__(
        self,
        model: BaseChatModel,
        toolkit: BaseEnvSetupToolkit,
        language: str,
        max_iterations: Optional[int] = None,
    ):
        self.toolkit = toolkit
        self.model = model
        self.language = language
        self._max_iterations = max_iterations

    @property
    def max_iterations(self) -> Optional[int]:
        if self._max_iterations is None:
            return None
        return 2 * (2 * self._max_iterations + 1)

    @property
    def configurable_config(self) -> InstallamaticConfigurable:  # type: ignore[override]
        search_tools = self.toolkit.get_tools(stage="search")
        build_tools = self.toolkit.get_tools(stage="build")
        submit_shell_script_tool = self.toolkit.get_tools(stage="submit_shell_script")[0]
        submit_summary_tool, get_directory_contents_tool = None, None
        for tool in build_tools:
            if tool.name == "submit_summary":
                submit_summary_tool = tool
            if tool.name == "get_directory_contents":
                get_directory_contents_tool = tool
        if submit_summary_tool is None:
            raise ValueError("Could not find submit_summary tool among tools for build stage.")
        if get_directory_contents_tool is None:
            raise ValueError("Could not find get_directory_contents tool among tools for build stage.")

        return {
            "search": {
                "model": self.model.bind_tools(search_tools),
                "get_directory_contents_tool": get_directory_contents_tool,
                "language": self.language,
            },
            "build": {
                "model": self.model.bind_tools(build_tools),
                "model_w_submit_summary_tool": self.model.bind_tools(
                    [submit_summary_tool], tool_choice="submit_summary"
                ),
                "model_w_submit_shell_script_tool": self.model.bind_tools(
                    [submit_shell_script_tool], tool_choice="submit_shell_script"
                ),
                "language": self.language,
            },
        }

    @property
    def commands_history(self) -> List[CommandExecutionResult]:
        return self.toolkit.commands_history

    def get_agent(self) -> CompiledGraph:
        search_tools = self.toolkit.get_tools(stage="search")
        build_tools = self.toolkit.get_tools(stage="build")
        graph = StateGraph(InstallamaticState)
        search_graph = InstallamaticSearchGraph.get_graph(tools=search_tools)
        build_graph = InstallamaticBuildGraph.get_graph(tools=build_tools)
        graph.add_node("search", search_graph)
        graph.add_node("build", build_graph)

        graph.set_entry_point("search")
        graph.add_edge("search", "build")
        graph.add_edge("build", END)
        return graph.compile()

    def construct_initial_state(self, repository: str, revision: str, *args, **kwargs) -> InstallamaticState:
        return {"repository": repository, "stage": "search"}

    @staticmethod
    def process_update_for_trajectory(update: InstallamaticUpdate, *args, **kwargs) -> InstallamaticTrajectoryEntry:
        timestamp = update["timestamp"]
        if "agent" in update:
            node = "agent"
            messages = update["agent"].get("messages", [])
            return {
                "timestamp": timestamp,
                "node": node,
                "messages": [message_to_info(message) for message in messages],
            }
        elif "tools" in update:
            node = "tools"
            messages = update["tools"].get("messages", [])
            return {
                "timestamp": timestamp,
                "node": node,
                "messages": [message_to_info(message) for message in messages],
            }
        elif "add_documentation" in update:
            node = "add_documentation"
            return {
                "timestamp": timestamp,
                "node": node,
                "documentation": list(update[node].get("documentation", [])),
            }
        elif "encourage_submit_documentation" in update:
            node = "encourage_submit_documentation"
            messages = update["encourage_submit_documentation"].get("messages", [])
            return {
                "timestamp": timestamp,
                "node": node,
                "messages": [message_to_info(message) for message in messages],
            }
        elif "init_state" in update:
            node = "init_state"
            return {
                "timestamp": timestamp,
                "stage": update[node].get("stage"),
            }
        elif "submit_summary" in update:
            node = "submit_summary"
            return {
                "timestamp": timestamp,
                "node": node,
                "summary": update[node].get("summary"),
            }
        elif "force_submit_summary_call" in update:
            node = "force_submit_summary_call"
            messages = update["force_submit_summary_call"].get("messages", [])
            return {
                "timestamp": timestamp,
                "node": node,
                "messages": [message_to_info(message) for message in messages],
            }
        elif "generate_shell_script" in update:
            node = "generate_shell_script"
            return {
                "timestamp": timestamp,
                "node": node,
                "shell_script": update[node].get("shell_script"),
            }
        elif "search" in update:
            node = "search"
            return {
                "timestamp": timestamp,
                "node": node,
                "documentation": list(update[node].get("documentation", [])),
                "stage": update[node].get("stage"),
            }
        elif "build" in update:
            node = "build"
            return {
                "timestamp": timestamp,
                "node": node,
                "shell_script": update[node].get("shell_script"),
                "stage": update[node].get("stage"),
            }

        raise RuntimeError(
            f"Expected the update to come from one of the graph nodes, but got {set(update.keys()) - {'timestamp'}}."
        )
