from operator import add
from typing import Annotated, Any, Dict, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class MultiAttemptState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add]
    turn: int
    tools_kwargs: Dict[str, Any]
