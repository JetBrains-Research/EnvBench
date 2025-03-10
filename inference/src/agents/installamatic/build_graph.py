from typing import List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode

from .prompts import get_installamatic_build_prompt, get_installamatic_generate_shell_script_prompt
from .state_schema import InstallamaticBuildConfigurable, InstallamaticBuildState


class InstallamaticBuildGraph:
    @staticmethod
    async def get_initial_prompt(state: InstallamaticBuildState, config: RunnableConfig) -> List[BaseMessage]:
        return await get_installamatic_build_prompt(state, config)

    @staticmethod
    async def get_shell_script_prompt(state: InstallamaticBuildState, config: RunnableConfig) -> List[BaseMessage]:
        return await get_installamatic_generate_shell_script_prompt(state, config)

    @staticmethod
    async def init_state(state: InstallamaticBuildState, config: RunnableConfig) -> InstallamaticBuildState:
        return {"stage": "build"}

    @staticmethod
    async def agent(state: InstallamaticBuildState, config: RunnableConfig) -> InstallamaticBuildState:
        configurable: InstallamaticBuildConfigurable = config.get("configurable", {}).get("build", {})
        model = configurable["model"]

        messages = []
        if not state.get("messages"):
            initial_messages = await InstallamaticBuildGraph.get_initial_prompt(state, config)
            messages.extend(initial_messages)
        else:
            messages.extend(state.get("messages", []))

        response = await model.ainvoke(messages, config=config)

        return {"messages": messages + [response] if not state.get("messages") else [response]}

    @staticmethod
    async def force_submit_summary_call(
        state: InstallamaticBuildState, config: RunnableConfig
    ) -> InstallamaticBuildState:
        configurable: InstallamaticBuildConfigurable = config.get("configurable", {}).get("build", {})
        model = configurable["model_w_submit_summary_tool"]

        messages = []
        if not state.get("messages"):
            initial_messages = await InstallamaticBuildGraph.get_initial_prompt(state, config)
            messages.extend(initial_messages)
        else:
            messages.extend(state.get("messages", []))

        response = await model.ainvoke(messages, config=config)

        return {"messages": messages + [response] if not state.get("messages") else [response]}

    @staticmethod
    async def submit_summary(state: InstallamaticBuildState, config: RunnableConfig) -> InstallamaticBuildState:
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            last_message = messages[-1]
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "submit_summary":
                    return {"summary": tool_call["args"]["summary"]}

        raise ValueError("submit_summary expects submit_summary tool call in the last message.")

    @staticmethod
    async def generate_shell_script(state: InstallamaticBuildState, config: RunnableConfig) -> InstallamaticBuildState:
        configurable: InstallamaticBuildConfigurable = config.get("configurable", {}).get("build", {})
        model = configurable["model_w_submit_shell_script_tool"]

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

        initial_messages = await InstallamaticBuildGraph.get_shell_script_prompt(state, config)
        messages.extend(initial_messages)
        response = await model.ainvoke(messages, config=config)

        shell_script: Optional[str] = None
        if isinstance(response, AIMessage):
            for tool_call in response.tool_calls:
                if tool_call["name"] == "submit_shell_script":
                    shell_script = tool_call["args"]["script"]
                    break

        if shell_script is None:
            raise ValueError("generate_shell_script node expects submit_shell_script tool call in the model response.")

        return {"shell_script": shell_script}

    @staticmethod
    def route_after_agent(
        state: InstallamaticBuildState, config: RunnableConfig
    ) -> Literal["tools", "submit_summary", "force_submit_summary_call"]:
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            last_message = messages[-1]
            assert isinstance(last_message, AIMessage), "route_after_agent edge expects to receive AIMessage."
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "submit_summary":
                    # need to proceed to submitting summary
                    return "submit_summary"
            # need to execute tool
            return "tools"
        # need to proceed to submitting summary, but the model didn't actually call this tool
        # => let's use forced function calling
        return "force_submit_summary_call"

    @staticmethod
    def get_graph(tools: List[BaseTool]) -> CompiledGraph:
        graph = StateGraph(InstallamaticBuildState)

        graph.add_node("init_state", InstallamaticBuildGraph.init_state)
        graph.add_node("agent", InstallamaticBuildGraph.agent)
        tool_node = ToolNode(tools)
        graph.add_node("tools", tool_node)
        graph.add_node("force_submit_summary_call", InstallamaticBuildGraph.force_submit_summary_call)
        graph.add_node("submit_summary", InstallamaticBuildGraph.submit_summary)
        graph.add_node("generate_shell_script", InstallamaticBuildGraph.generate_shell_script)

        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "agent")
        graph.add_conditional_edges("agent", InstallamaticBuildGraph.route_after_agent)
        graph.add_edge("tools", "agent")
        graph.add_edge("force_submit_summary_call", "submit_summary")
        graph.add_edge("submit_summary", "generate_shell_script")
        graph.add_edge("generate_shell_script", END)
        return graph.compile()
