from textwrap import dedent
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import convert_to_messages
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from .state_schema import MultiAttemptState


def create_multi_attempt_workflow(
    model: BaseChatModel,
    max_iterations: Optional[int] = None,
) -> CompiledStateGraph:
    """Create a compiled multi-attempt workflow graph.

    Args:
        model: The language model to use for generating responses
        max_iterations: Maximum number of iterations before stopping

    Returns:
        A compiled graph ready for execution
    """
    default_max_turns = max_iterations or 2

    def initialize_state(state: MultiAttemptState) -> MultiAttemptState:
        """Initialize the state with default values if not present."""
        return {
            "messages": convert_to_messages(state.get("messages", [])),
            "turn": state.get("turn", 1),
        }

    def call_model(state: MultiAttemptState) -> MultiAttemptState:
        new_message = model.invoke(state["messages"])
        return {
            "messages": [new_message],
        }

    def next_turn(state: MultiAttemptState) -> MultiAttemptState:
        return {
            "messages": [
                HumanMessage(
                    content=dedent("""
                        Review your previous response and check:

                        1. Reasoning is sound and tailored to this repository's context
                        2. Script is enclosed in ```bash``` fences
                        3. Script includes:
                           - Correct Python version installation
                           - Project dependencies
                           - Required system packages
                        4. Script follows best practices:
                           - Uses -y flags for non-interactive mode
                           - Uses pyenv install -f where required
                           - Avoids sudo commands
                           - Doesn't install tools already available in the system

                        If you find errors or omissions, resend only the corrected script.
                        Otherwise, resend only the final script.
                    """)
                )
            ],
            "turn": state["turn"] + 1,
        }

    def should_continue(state: MultiAttemptState) -> Literal["next_turn", "__end__"]:
        if state["turn"] < default_max_turns:
            return "next_turn"
        return "__end__"

    # Create the graph
    workflow = StateGraph(MultiAttemptState)

    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("model", call_model)
    workflow.add_node("next_turn", next_turn)

    # Add edges
    workflow.add_edge("initialize", "model")
    workflow.add_conditional_edges(
        "model",
        should_continue,
    )
    workflow.add_edge("next_turn", "model")

    # Set entry point
    workflow.set_entry_point("initialize")

    return workflow.compile()
