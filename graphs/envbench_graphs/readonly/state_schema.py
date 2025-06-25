"""Bash read-only workflow graph implementation."""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class EnvSetupReadOnlyState(TypedDict, total=False):
    """State schema for bash read-only workflow."""

    messages: Annotated[List[BaseMessage], add_messages]
    turn: int
    shell_script: Optional[str]
    tools_kwargs: Dict[str, Any]
