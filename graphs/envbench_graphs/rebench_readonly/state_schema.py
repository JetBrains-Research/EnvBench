from typing import Dict, List, TypedDict

from langchain_core.messages import BaseMessage


class RebenchReadonlyState(TypedDict):
    """State schema for the rebench-readonly workflow."""

    # Messages
    messages: List[BaseMessage]

    # Repository info
    tools_kwargs: Dict
