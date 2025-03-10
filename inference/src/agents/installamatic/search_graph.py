from typing import List, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode

from .prompts import get_installamatic_search_prompt
from .state_schema import InstallamaticSearchConfigurable, InstallamaticSearchState


class InstallamaticSearchGraph:
    @staticmethod
    async def get_initial_prompt(state: InstallamaticSearchState, config: RunnableConfig) -> List[BaseMessage]:
        return await get_installamatic_search_prompt(state, config)

    @staticmethod
    async def encourage_submit_documentation(
        state: InstallamaticSearchState, config: RunnableConfig
    ) -> InstallamaticSearchState:
        messages = state.get("messages", [])
        unanswered_tool_calls = set()
        messages_to_add: List[BaseMessage] = []
        for message in messages:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    unanswered_tool_calls.add(tool_call["id"])
            if isinstance(message, ToolMessage):
                unanswered_tool_calls.remove(message.tool_call_id)

        for tool_call_id in unanswered_tool_calls:
            messages_to_add.append(
                ToolMessage(content="Sorry, let's add some documentation first.", tool_call_id=tool_call_id)
            )

        messages_to_add.append(
            HumanMessage(
                "Please, submit documentation before calling `finished_search` tool or answering without a tool call."
            )
        )
        return {"messages": messages_to_add}

    @staticmethod
    async def agent(state: InstallamaticSearchState, config: RunnableConfig) -> InstallamaticSearchState:
        configurable: InstallamaticSearchConfigurable = config.get("configurable", {}).get("search", {})
        model = configurable["model"]

        messages = []
        if not state.get("messages"):
            initial_messages = await InstallamaticSearchGraph.get_initial_prompt(state, config)
            messages.extend(initial_messages)
        else:
            messages.extend(state.get("messages", []))

        response = await model.ainvoke(messages, config=config)

        return {"messages": messages + [response] if not state.get("messages") else [response]}

    @staticmethod
    async def add_documentation(state: InstallamaticSearchState, config: RunnableConfig) -> InstallamaticSearchState:
        messages = state.get("messages", [])
        cur_documentation = state.get("documentation", set())
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            last_message = messages[-1]
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "submit_documentation":
                    cur_documentation.add(tool_call["args"]["file"])
        return {"documentation": cur_documentation}

    @staticmethod
    def route_after_agent(
        state: InstallamaticSearchState, config: RunnableConfig
    ) -> Literal["tools", "add_documentation", "encourage_submit_documentation", "__end__"]:
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            last_message = messages[-1]
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "finished_search":
                    # need to end because finished search tool is called
                    if not state.get("documentation"):
                        return "encourage_submit_documentation"
                    return "__end__"
                if tool_call["name"] == "submit_documentation":
                    # need to add documentation because submit documentation tool is called
                    return "add_documentation"
            # need to execute tool
            return "tools"
        # need to end because no tool calls
        if not state.get("documentation"):
            return "encourage_submit_documentation"
        return "__end__"

    @staticmethod
    def get_graph(tools: List[BaseTool]) -> CompiledGraph:
        graph = StateGraph(InstallamaticSearchState)

        graph.add_node("agent", InstallamaticSearchGraph.agent)
        tool_node = ToolNode(tools)
        graph.add_node("tools", tool_node)
        graph.add_node("add_documentation", InstallamaticSearchGraph.add_documentation)
        graph.add_node("encourage_submit_documentation", InstallamaticSearchGraph.encourage_submit_documentation)

        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", InstallamaticSearchGraph.route_after_agent)
        graph.add_edge("add_documentation", "tools")
        graph.add_edge("encourage_submit_documentation", "agent")
        graph.add_edge("tools", "agent")

        return graph.compile()
