from typing import Awaitable, Callable, Literal, Optional, Tuple, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph.state import CompiledStateGraph, StateGraph

from .state_schema import TryAgainState


def create_try_again_workflow(
    model: BaseChatModel,
    run_reward_func: Union[
        Callable[[BaseMessage], Awaitable[Tuple[float, str]]],
        Callable[[BaseMessage, TryAgainState], Awaitable[Tuple[float, str]]]
    ],
    reward_score_for_pass: float = 1.0,
    max_iterations: Optional[int] = 2,
    pass_state: bool = False,
) -> CompiledStateGraph:
    """Create a compiled try-again graph.
    (iterate until reward score is >= reward_score_for_pass or max_iterations is reached)

    Args:
        model: The language model to use for generating responses
        run_reward_func: An async function that takes a message (and optionally state) and returns a reward score and text feedback
        reward_score_for_pass: The reward score threshold for stopping the iterations
        max_iterations: Maximum number of iterations before stopping
        pass_state: Whether to pass the state to the reward function in addition to the message

    Returns:
        A compiled graph ready for execution
    """

    def should_continue(state: TryAgainState) -> Literal["run_turn", "__end__"]:
        if state["turn"] >= state["max_turns"]:
            return "__end__"

        # we didn't add feedback => we passed => yay
        if isinstance(state["messages"][-1], AIMessage):
            return "__end__"

        return "run_turn"

    def initialize_state(state: TryAgainState) -> TryAgainState:
        """Initialize the state with default values if not present."""
        return {
            "turn": state.get("turn", 0),
            "max_turns": state.get("max_turns", max_iterations),
            "tools_kwargs": state.get("tools_kwargs", {}),
        }

    async def run_turn(state: TryAgainState) -> TryAgainState:
        new_message = await model.ainvoke(state["messages"])

        if pass_state:
            reward_score, feedback_text = await run_reward_func(new_message, state)
        else:
            reward_score, feedback_text = await run_reward_func(new_message)

        new_message.additional_kwargs["reward_score"] = reward_score
        feedback_messages = []
        if reward_score < reward_score_for_pass:
            feedback_messages.append(HumanMessage(content=feedback_text))
        return {"messages": [new_message] + feedback_messages,
                "turn": state["turn"] + 1}

    # Create the graph
    workflow = StateGraph(TryAgainState)

    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("run_turn", run_turn)

    # Add edges
    workflow.add_edge("initialize", "run_turn")
    workflow.add_conditional_edges("run_turn", should_continue)

    # Set entry point
    workflow.set_entry_point("initialize")

    return workflow.compile()
