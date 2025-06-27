from typing import Awaitable, Callable, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import END, StateGraph
from langgraph.prebuilt import create_react_agent

from envbench_graphs.rebench_setup.graph import default_execute_bash_command

from .prompts import (
    BASH_SCRIPT_NO_SUMMARY_PROMPT,
    BASH_SCRIPT_PROMPT,
    REACT_AGENT_PROMPT,
    SUMMARIZATION_PROMPT,
    python_dockerfile,
)
from .state_schema import RebenchReadonlyState


def truncate(text: str, max_length) -> str:
    """Truncate text to a maximum length."""
    if max_length < 0 or len(text) <= max_length:
        return text
    return text[:max_length] + f"\n[truncated to {max_length} characters]"


def create_rebench_readonly_workflow(
    model: BaseChatModel,
    exploration_model: Optional[BaseChatModel] = None,
    do_summarization: bool = False,
    max_turns: int = 10,
    entrypoint_node: str = "initialize",
    execute_bash_command: Optional[Callable[[str, str], Awaitable[Dict[str, str]]]] = None,
    max_length: int = 8000,
) -> CompiledGraph:
    """
    Create a compiled workflow graph for rebench-readonly operations.

    Args:
        model: The model to use for the workflow (actor)
        exploration_model: The model to use for the exploration (explorer + summarizer)
        do_summarization: Whether to do summarization.
        max_turns: The maximum number of turns for the ReAct agent.
        entrypoint_node: The entrypoint node for the workflow.
        execute_bash_command: The function to use to execute bash commands.
        max_length: The maximum length of the stdout and stderr of the bash command.
    """

    if execute_bash_command is None:
        execute_bash_command = default_execute_bash_command

    # Use the same model for exploration if none provided
    if exploration_model is None:
        exploration_model = model

    def initialize_state(state: RebenchReadonlyState) -> RebenchReadonlyState:
        """Initialize the state with default values and add ReAct agent prompt."""
        repo_name = state["tools_kwargs"].get("repository", "unknown")
        message = HumanMessage(content=REACT_AGENT_PROMPT.format(repo_name=repo_name))

        return {
            "messages": [message],
            "tools_kwargs": state.get("tools_kwargs", {}),
        }

    def limit_turns(state: RebenchReadonlyState) -> RebenchReadonlyState:
        """Pre-model hook to limit the number of turns for the ReAct agent."""
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        if len(ai_messages) >= max_turns:
            stop_message = AIMessage(content="Sorry, need more steps to process this request.")
            return state | {"messages": state["messages"] + [stop_message]}
        return state

    # Create bash command tool
    def create_bash_tool(state: RebenchReadonlyState):
        @tool
        async def bash_tool(command: str) -> str:
            """Execute a bash command in the repository."""
            repository = state["tools_kwargs"].get("repository", "unknown").replace("/", "__")
            result = await execute_bash_command(command, repository)
            return f"Exit code: {result['exit_code']}\nstdout: {truncate(result['stdout'], max_length)}\nstderr: {truncate(result['stderr'], max_length)}"

        return bash_tool

    async def start_exploration(state: RebenchReadonlyState) -> RebenchReadonlyState:
        """Start the exploration with the ReAct agent."""
        # Create the ReAct agent with the bash tool
        bash_tool = create_bash_tool(state)
        agent = create_react_agent(model=exploration_model, tools=[bash_tool], pre_model_hook=limit_turns)

        # Run the agent with existing messages
        result = await agent.ainvoke(state)

        # Ensure we return a proper RebenchReadonlyState
        return {
            "messages": result.get("messages", []),
            "tools_kwargs": result.get("tools_kwargs", state["tools_kwargs"]),
        }

    async def optional_summarization(state: RebenchReadonlyState) -> RebenchReadonlyState:
        """Optional summarization step."""
        repo_name = state["tools_kwargs"].get("repository", "unknown")

        if do_summarization:
            # Extract exploration messages for summarization
            exploration_messages = state["messages"]

            message = HumanMessage(content=SUMMARIZATION_PROMPT.format(repo_name=repo_name))
            response = await exploration_model.ainvoke(exploration_messages + [message])

            # Add the bash script generation message
            bash_message = HumanMessage(
                content=BASH_SCRIPT_PROMPT.format(
                    repo_name=repo_name, summary=str(response.content), dockerfile=python_dockerfile
                )
            )

            return {
                "messages": [message, response, bash_message],  # Summary messages + bash generation message
                "tools_kwargs": state["tools_kwargs"],
            }
        else:
            # Add the bash script generation message
            message = HumanMessage(
                content=BASH_SCRIPT_NO_SUMMARY_PROMPT.format(repo_name=repo_name, dockerfile=python_dockerfile)
            )
            return state | {"messages": state["messages"] + [message]}

    async def generate_bash_script(state: RebenchReadonlyState) -> RebenchReadonlyState:
        """Generate the final bash script."""
        response = await model.ainvoke(state["messages"])

        return {
            "messages": state["messages"] + [response],
            "tools_kwargs": state["tools_kwargs"],
        }

    # Create the graph
    workflow = StateGraph(RebenchReadonlyState)

    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("start_exploration", start_exploration)
    workflow.add_node("optional_summarization", optional_summarization)
    workflow.add_node("generate_bash_script", generate_bash_script)

    # Add edges
    workflow.add_edge("initialize", "start_exploration")
    workflow.add_edge("start_exploration", "optional_summarization")
    workflow.add_edge("optional_summarization", "generate_bash_script")
    workflow.add_edge("generate_bash_script", END)

    # Set entry point
    workflow.set_entry_point(entrypoint_node)

    return workflow.compile()
