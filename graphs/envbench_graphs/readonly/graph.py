from typing import List, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode

from .prompts import get_readonly_prompt
from .state_schema import EnvSetupReadOnlyState


def create_read_only_workflow(
    model: BaseChatModel,
    tools: List[BaseTool],
    submit_shell_script_tool: BaseTool,
    max_iterations: Optional[int] = None,
) -> CompiledGraph:
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
            "turn": state.get("turn", 1),
        }

    async def call_model(state: EnvSetupReadOnlyState) -> EnvSetupReadOnlyState:
        new_message = await model.ainvoke(state["messages"])
        return {
            "messages": [new_message],
        }

    async def call_tools(state: EnvSetupReadOnlyState, config: RunnableConfig) -> EnvSetupReadOnlyState:
        # pass tools_kwargs from state to tool node
        tools_kwargs = state.get("tools_kwargs", {})
        if "configurable" in config:
            config["configurable"]["tools_kwargs"] = tools_kwargs
        else:
            config["configurable"] = {"tools_kwargs": tools_kwargs}
        print("[DEBUG] tools_kwargs:", tools_kwargs)
        print("[DEBUG] config:", config)
        response = await tool_node.ainvoke(state["messages"], config=config)
        return {"messages": response, "turn": state["turn"] + 1}

    async def force_submit_shell_script_tool_call(
        state: EnvSetupReadOnlyState, config: RunnableConfig
    ) -> EnvSetupReadOnlyState:
        model_w_submit_shell_script_tool = model.bind_tools(
            [submit_shell_script_tool], tool_choice="submit_shell_script"
        )

        messages = state.get("messages", [])
        unanswered_tool_calls = set()
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

        response = await model_w_submit_shell_script_tool.ainvoke(messages, config=config)

        return {"messages": [response]}

    async def submit_shell_script(state: EnvSetupReadOnlyState, config: RunnableConfig) -> EnvSetupReadOnlyState:
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            last_message = messages[-1]
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "submit_shell_script":
                    return {"shell_script": tool_call["args"]["script"]}

        raise ValueError("submit_shell_script expects submit_shell_script tool call in the last message.")

    def route_after_call_model(
        state: EnvSetupReadOnlyState, config: RunnableConfig
    ) -> Literal["tools", "submit_shell_script", "force_submit_shell_script_tool_call"]:
        if max_iterations and state["turn"] >= max_iterations:
            return "force_submit_shell_script_tool_call"

        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            last_message = messages[-1]
            assert isinstance(last_message, AIMessage), "route_after_call_model edge expects to receive AIMessage."
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "submit_shell_script":
                    # submitting shell script
                    return "submit_shell_script"
            # need to execute tool
            # but we've reached the max number of turns
            # the model didn't actually submit the script => let's force it
            if max_iterations and state["turn"] >= max_iterations:
                return "force_submit_shell_script_tool_call"
            return "tools"
        # no tool calls, so we end, but no submitting scripts => let's force it
        return "force_submit_shell_script_tool_call"

    graph = StateGraph(EnvSetupReadOnlyState)

    graph.add_node("init_state", initialize_state)
    graph.add_node("model", call_model)

    graph.add_node("tools", call_tools)
    graph.add_node("force_submit_shell_script_tool_call", force_submit_shell_script_tool_call)
    graph.add_node("submit_shell_script", submit_shell_script)

    graph.set_entry_point("init_state")
    graph.add_edge("init_state", "model")
    graph.add_conditional_edges("model", route_after_call_model)
    graph.add_edge("tools", "model")
    graph.add_edge("force_submit_shell_script_tool_call", "submit_shell_script")
    graph.add_edge("submit_shell_script", END)
    return graph.compile()
