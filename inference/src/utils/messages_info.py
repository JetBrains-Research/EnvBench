from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.messages.ai import UsageMetadata


class MessageType(Enum):
    system = "system"
    human = "human"
    ai = "ai"
    tool = "tool"


class BaseMessageContent(TypedDict):
    content: Union[str, List[Union[str, Dict]]]


class AIMessageContent(BaseMessageContent):
    tool_calls: List[ToolCall]
    usage_metadata: Optional[UsageMetadata]


class ToolMessageContent(BaseMessageContent):
    tool_call_id: str


class MessageInfo(TypedDict, total=False):
    message_type: str
    message_content: BaseMessageContent | AIMessageContent | ToolMessageContent
    response_metadata: Optional[Dict[str, Any]]


def message_to_info(message: BaseMessage) -> MessageInfo:
    message_type = MessageType(message.type)

    if isinstance(message, SystemMessage):
        return {
            "message_type": message_type.value,
            "message_content": {"content": message.content},
            "response_metadata": message.response_metadata,
        }
    if isinstance(message, HumanMessage):
        return {
            "message_type": message_type.value,
            "message_content": {"content": message.content},
            "response_metadata": message.response_metadata,
        }
    if isinstance(message, AIMessage):
        return {
            "message_type": message_type.value,
            "message_content": {
                "content": message.content,
                "tool_calls": message.tool_calls,
                "usage_metadata": message.usage_metadata,
            },
            "response_metadata": message.response_metadata,
        }
    if isinstance(message, ToolMessage):
        return {
            "message_type": message_type.value,
            "message_content": {"content": message.content, "tool_call_id": message.tool_call_id},
            "response_metadata": message.response_metadata,
        }

    raise RuntimeError(f"Unknown message type {type(message)}.")
