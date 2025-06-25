from operator import add
from typing import Annotated, Any, Dict, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class ShellcheckState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add]
    turn: Annotated[int, "Current turn number"]
    max_turns: Annotated[int, "Maximum number of turns allowed"]
    should_continue: bool | None
    script: str | None
    tools_kwargs: Dict[str, Any]
