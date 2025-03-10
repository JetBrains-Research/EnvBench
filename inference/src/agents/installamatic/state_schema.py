from typing import Annotated, List, Optional, Set, TypedDict, Union

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import add_messages

from src.utils.messages_info import MessageInfo


# ====================
# =      SEARCH      =
# ====================
class InstallamaticSearchConfigurable(TypedDict):
    model: Runnable[LanguageModelInput, BaseMessage]
    get_directory_contents_tool: BaseTool
    language: str


class InstallamaticSearchState(TypedDict, total=False):
    # main graph fields
    repository: str
    documentation: Set[str]
    shell_script: str
    stage: str
    # subgraph fields
    messages: Annotated[List[BaseMessage], add_messages]


# ====================
# =       BUILD      =
# ====================
class InstallamaticBuildConfigurable(TypedDict):
    model_w_submit_shell_script_tool: Runnable[LanguageModelInput, BaseMessage]
    model_w_submit_summary_tool: Runnable[LanguageModelInput, BaseMessage]
    model: Runnable[LanguageModelInput, BaseMessage]
    language: str


class InstallamaticBuildState(TypedDict, total=False):
    # main graph fields
    repository: str
    documentation: Set[str]
    shell_script: str
    stage: str
    # subgraph fields
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str


# ====================
# =    MAIN GRAPH    =
# ====================
class InstallamaticConfigurable(TypedDict):
    search: InstallamaticSearchConfigurable
    build: InstallamaticBuildConfigurable


class InstallamaticState(TypedDict, total=False):
    repository: str
    documentation: Set[str]
    shell_script: str
    stage: str


class InstallamaticUpdate(TypedDict):
    agent: Union[InstallamaticSearchState, InstallamaticBuildState]
    tools: Union[InstallamaticSearchState, InstallamaticBuildState]
    add_documentation: InstallamaticSearchState
    encourage_submit_documentation: InstallamaticSearchState
    init_state: InstallamaticBuildState
    submit_summary: InstallamaticBuildState
    force_submit_summary_call: InstallamaticBuildState
    generate_shell_script: InstallamaticBuildState
    search: InstallamaticState
    build: InstallamaticState
    timestamp: str


class InstallamaticTrajectoryEntry(TypedDict, total=False):
    node: str
    stage: Optional[str]
    documentation: List[str]
    summary: Optional[str]
    shell_script: Optional[str]
    messages: List[MessageInfo]
    timestamp: str
