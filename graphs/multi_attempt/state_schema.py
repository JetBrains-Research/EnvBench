from operator import add
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class MultiAttemptState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add]
    turn: int
