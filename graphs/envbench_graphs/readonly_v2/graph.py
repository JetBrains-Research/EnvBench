import re
from typing import List, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from .prompts import get_readonly_prompt
from .state_schema import EnvSetupReadOnlyState


def parse_script_from_content(content: str) -> Optional[str]:
    """Extract script from content using the ```bash pattern."""
    if not content:
        return None

    # Look for the pattern: ```bash\n...script content...\n```
    pattern = r"```bash\s*\n(.*?)\n\s*```"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def create_read_only_workflow(
    model: BaseChatModel,
    tools: List[BaseTool],
    max_iterations: Optional[int] = None,
    max_script_generation_attempts: int = 3,
) -> CompiledStateGraph:
    """Create a read-only workflow that uses ```bash``` pattern for script generation."""
    model = model.bind_tools(tools)
    tool_node = ToolNode(tools)

    def get_initial_prompt(state: EnvSetupReadOnlyState, config: RunnableConfig) -> List[BaseMessage]:
        return get_readonly_prompt(state, config)

    def initialize_state(state: EnvSetupReadOnlyState, config: RunnableConfig) -> EnvSetupReadOnlyState:
        """Initialize the state with default values if not present."""
        messages = state.get("messages", [])
        if not messages:
            messages = get_initial_prompt(state, config)
        return {
            "messages": convert_to_messages(messages),
            "tools_kwargs": state.get("tools_kwargs", [{}]),
            "turn": state.get("turn", 1),
        }

    async def call_model(state: EnvSetupReadOnlyState) -> EnvSetupReadOnlyState:
        new_message = await model.ainvoke(state["messages"])
        return {
            "messages": [new_message],
        }

    async def call_tools(state: EnvSetupReadOnlyState, config: RunnableConfig) -> EnvSetupReadOnlyState:
        response = await tool_node.ainvoke(state, config)
        return {"messages": response["messages"], "turn": state["turn"] + 1}

    async def force_script_generation(state: EnvSetupReadOnlyState, config: RunnableConfig) -> EnvSetupReadOnlyState:
        """Force the model to generate a script using the script pattern with retries."""
        messages = state.get("messages", [])
        unanswered_tool_calls = set()

        # Handle any unanswered tool calls
        for message in messages:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    unanswered_tool_calls.add(tool_call["id"])
            if isinstance(message, ToolMessage):
                unanswered_tool_calls.remove(message.tool_call_id)

        for tool_call_id in unanswered_tool_calls:
            messages.append(
                ToolMessage(
                    content="Sorry, you've already moved on to generating shell script.", tool_call_id=tool_call_id
                )
            )

        # Try generating script with retries
        for attempt in range(max_script_generation_attempts):
            # Add a message encouraging script generation
            if attempt == 0:
                messages.append(
                    HumanMessage(
                        content="Now you need to generate the final shell script. Please use ```bash\n...script content...\n``` pattern for the script."
                    )
                )
            else:
                messages.append(
                    HumanMessage(
                        content=f"The script pattern was not found in your previous response. Please try again (attempt {attempt + 1}/{max_script_generation_attempts}). Use ```bash\n...script content...\n``` pattern for the script."
                    )
                )

            response = await model.ainvoke(messages, config=config)
            shell_script = parse_script_from_content(response.content if response.content else "")

            if shell_script:
                return {"messages": [response], "shell_script": shell_script}

            # Add the response to messages for the next iteration
            messages.append(response)

        # Return the last response even if no script was found
        return {"messages": [response], "shell_script": shell_script}

    async def extract_shell_script(state: EnvSetupReadOnlyState, config: RunnableConfig) -> EnvSetupReadOnlyState:
        """Extract shell script from the last message."""
        messages = state.get("messages", [])
        shell_script = None

        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and last_message.content:
                shell_script = parse_script_from_content(last_message.content)

        return {"shell_script": shell_script}

    def route_after_call_model(
        state: EnvSetupReadOnlyState, config: RunnableConfig
    ) -> Literal["tools", "extract_shell_script", "force_script_generation"]:
        """Route the workflow after model call based on the response."""
        if max_iterations and state["turn"] >= max_iterations:
            return "force_script_generation"

        messages = state.get("messages", [])
        if not messages:
            return "force_script_generation"

        last_message = messages[-1]

        # Check if the last message contains a script pattern
        if isinstance(last_message, AIMessage) and last_message.content:
            if parse_script_from_content(last_message.content):
                return "extract_shell_script"

        # Check if there are tool calls to execute
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            # If we've reached max iterations, force script generation
            if max_iterations and state["turn"] >= max_iterations:
                return "force_script_generation"
            return "tools"

        # No tool calls and no script, force script generation
        return "force_script_generation"

    # Build the graph
    graph = StateGraph(EnvSetupReadOnlyState)

    graph.add_node("init_state", initialize_state)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_node("force_script_generation", force_script_generation)
    graph.add_node("extract_shell_script", extract_shell_script)

    # Define edges
    graph.set_entry_point("init_state")
    graph.add_edge("init_state", "model")
    graph.add_conditional_edges("model", route_after_call_model)
    graph.add_edge("tools", "model")
    graph.add_edge("force_script_generation", END)
    graph.add_edge("extract_shell_script", END)

    return graph.compile()
