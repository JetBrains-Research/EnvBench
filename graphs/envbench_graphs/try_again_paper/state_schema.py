from typing import Annotated, Any, Dict, Sequence, TypedDict, List
from operator import add
from langchain_core.messages import BaseMessage


class TryAgainState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add]
    turn: Annotated[int, "Current turn number"]
    max_turns: Annotated[int, "Maximum number of turns allowed"]
    tools_kwargs: Dict[str, Any]
