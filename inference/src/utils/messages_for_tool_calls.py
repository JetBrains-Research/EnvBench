import uuid
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool


def create_messages_for_tool_call(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_result: str,
) -> List[BaseMessage]:
    tool_call_id = uuid.uuid4().hex
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": tool.name,
                    "args": args,
                    "id": f"{tool.name}_fake_{tool_call_id}",
                }
            ],
        ),
        ToolMessage(content=tool_result, tool_call_id=f"{tool.name}_fake_{tool_call_id}"),
    ]
