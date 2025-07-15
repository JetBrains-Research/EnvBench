import re
from typing import Awaitable, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph.state import END, CompiledStateGraph, StateGraph

from .state_schema import ShellcheckState


def create_shellcheck_workflow(
    model: BaseChatModel,
    run_shellcheck_func: Callable[..., Awaitable[str]],
    max_iterations: Optional[int] = 2,
    pass_state: bool = False,
) -> CompiledStateGraph:
    """Create a compiled workflow graph for shellcheck operations.

    Args:
        model: The language model to use for generating responses
        run_shellcheck_func: An async function that takes a script string and returns shellcheck results
        max_iterations: Maximum number of iterations before stopping
        pass_state: Whether to pass the state to the run_shellcheck_func as a second argument

    Returns:
        A compiled graph ready for execution
    """

    def process_model_output(state: ShellcheckState) -> ShellcheckState:
        # Check if we should continue
        if state["turn"] >= state["max_turns"]:
            return {
                "should_continue": False,
            }

        # Get the last message from the model
        last_message = state["messages"][-1]
        content = str(last_message.content)

        # For QwQ models, remove reasoning
        if "qwq" in model.model_name.lower():
            content = re.sub(r"^.*?</think>", "", content, re.DOTALL).strip()

        # Find bash script blocks
        matches = re.findall(r"```bash(.*?)```", content, re.DOTALL)
        if not matches:
            return {"should_continue": False}

        return {
            "should_continue": True,
            "script": matches[0].strip(),
        }

    async def run_shellcheck(state: ShellcheckState) -> ShellcheckState:
        if not state["should_continue"]:
            return state

        script = state["script"] or ""

        if pass_state:
            result = await run_shellcheck_func(script, state)
        else:
            result = await run_shellcheck_func(script)

        return {"messages": [HumanMessage(content=result)], "turn": state["turn"] + 1}

    def initialize_state(state: ShellcheckState) -> ShellcheckState:
        """Initialize the state with default values if not present."""
        return {
            "turn": state.get("turn", 0),
            "max_turns": state.get("max_turns", max_iterations),
            "should_continue": None,
            "script": None,
            "tools_kwargs": state.get("tools_kwargs", {}),
        }

    def call_model(state: ShellcheckState) -> ShellcheckState:
        new_message = model.invoke(state["messages"])
        return {"messages": [new_message]}

    # Create the graph
    workflow = StateGraph(ShellcheckState)

    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("process_output", process_model_output)
    workflow.add_node("run_shellcheck", run_shellcheck)
    workflow.add_node("model", call_model)

    # Add edges
    workflow.add_edge("initialize", "model")
    workflow.add_edge("model", "process_output")
    workflow.add_conditional_edges(
        "process_output", lambda x: x["should_continue"], {True: "run_shellcheck", False: END}
    )
    workflow.add_edge("run_shellcheck", "model")

    # Set entry point
    workflow.set_entry_point("initialize")

    return workflow.compile()
