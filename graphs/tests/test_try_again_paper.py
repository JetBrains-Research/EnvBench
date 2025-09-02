
from envbench_graphs.try_again_paper.graph import create_try_again_workflow
from envbench_graphs.try_again_paper.state_schema import TryAgainState
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage, AIMessage
import pytest


@pytest.mark.asyncio
async def test_try_again_workflow_compilation():
    """Test that the try-again workflow compiles correctly."""

    async def mock_reward_func(message) -> tuple[float, str]:
        return 0.5, "Mock feedback"

    model = FakeListChatModel(responses=["Test response"])

    workflow = create_try_again_workflow(
        model=model, run_reward_func=mock_reward_func, reward_score_for_pass=1.0, max_iterations=3
    )

    assert workflow is not None


@pytest.mark.asyncio
async def test_try_again_workflow_max_iterations():
    """Test that the workflow stops at max iterations when reward never reaches threshold."""

    async def low_reward_func(message) -> tuple[float, str]:
        return 0.3, "Not good enough yet"

    model = FakeListChatModel(responses=["First attempt", "Second attempt", "Third attempt"])

    workflow = create_try_again_workflow(
        model=model, run_reward_func=low_reward_func, reward_score_for_pass=1.0, max_iterations=2
    )

    initial_state = {"messages": [HumanMessage(content="Initial task")]}

    result = await workflow.ainvoke(initial_state)

    assert result["turn"] == 2
    assert result["max_turns"] == 2
    rewards = [message.additional_kwargs["reward_score"] for message in result["messages"] if
               isinstance(message, AIMessage)]
    assert all(reward == 0.3 for reward in rewards)

    # expected: initial + (model + feedback) * 2 = 5 messages
    assert len(result["messages"]) == 5


@pytest.mark.asyncio
async def test_try_again_workflow_reward_threshold():
    """Test that the workflow stops when reward threshold is reached."""

    call_count = 0

    async def improving_reward_func(message) -> tuple[float, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 0.4, "First attempt: Needs improvement"
        else:
            return 1.2, "Second attempt: Excellent!"

    model = FakeListChatModel(responses=["First attempt response", "Second attempt response"])

    workflow = create_try_again_workflow(
        model=model,
        run_reward_func=improving_reward_func,
        reward_score_for_pass=1.0,
        max_iterations=5,  # High limit, should stop early due to reward
    )

    initial_state = {"messages": [HumanMessage(content="Initial task")]}

    result = await workflow.ainvoke(initial_state)

    assert result["turn"] == 2
    assert result["max_turns"] == 5
    assert len(result["messages"]) == 4
    rewards = [message.additional_kwargs["reward_score"] for message in result["messages"] if isinstance(message, AIMessage)]
    assert rewards[0] == 0.4
    assert rewards[1] == 1.2


@pytest.mark.asyncio
async def test_try_again_workflow_reward_feedback():
    """Test reward function integration and feedback handling."""

    reward_calls = []

    async def tracking_reward_func(message) -> tuple[float, str]:
        # Track all calls to verify message is passed correctly
        reward_calls.append(
            {
                "call_count": len(reward_calls) + 1,
                "message_content": message.content,
                "message_type": type(message).__name__,
            }
        )

        call_count = len(reward_calls)
        if call_count == 1:
            return 0.2, "Initial attempt: Low quality"
        elif call_count == 2:
            return 0.6, "Second attempt: Better but not there yet"
        else:
            return 1.1, "Final attempt: Excellent work!"

    model = FakeListChatModel(responses=["First model response", "Second model response", "Third model response"])

    workflow = create_try_again_workflow(
        model=model, run_reward_func=tracking_reward_func, reward_score_for_pass=1.0, max_iterations=5
    )

    initial_state = {
        "messages": [HumanMessage(content="Initial task")],
        "tools_kwargs": {"repo": "test-repo", "task": "setup"},
    }

    result = await workflow.ainvoke(initial_state)

    assert len(reward_calls) == 3
    assert reward_calls[0]["call_count"] == 1
    assert reward_calls[1]["call_count"] == 2
    assert reward_calls[2]["call_count"] == 3

    # Verify that AI messages are passed to reward function
    for call in reward_calls:
        assert call["message_type"] == "AIMessage"
        assert isinstance(call["message_content"], str)

    # Verify the model responses are what we expect
    assert reward_calls[0]["message_content"] == "First model response"
    assert reward_calls[1]["message_content"] == "Second model response"
    assert reward_calls[2]["message_content"] == "Third model response"

    assert result["turn"] == 3
    assert result["max_turns"] == 5
    rewards = [message.additional_kwargs["reward_score"] for message in result["messages"] if
               isinstance(message, AIMessage)]
    assert rewards == [0.2, 0.6, 1.1]

    messages = result["messages"]
    feedback_messages = [msg for msg in messages if isinstance(msg, HumanMessage) and "attempt" in msg.content.lower()]
    assert len(feedback_messages) == 2


@pytest.mark.asyncio
async def test_try_again_workflow_single_iteration_success():
    """Test workflow that succeeds on first try."""

    async def high_reward_func(message) -> tuple[float, str]:
        return 1.5, "Perfect on first try!"

    model = FakeListChatModel(responses=["Excellent response"])

    workflow = create_try_again_workflow(
        model=model, run_reward_func=high_reward_func, reward_score_for_pass=1.0, max_iterations=3
    )

    initial_state = {"messages": [HumanMessage(content="Simple task")]}

    result = await workflow.ainvoke(initial_state)

    assert result["turn"] == 1
    assert result["max_turns"] == 3
    rewards = [message.additional_kwargs["reward_score"] for message in result["messages"] if
               isinstance(message, AIMessage)]
    assert len(rewards) == 1
    assert rewards[0] == 1.5
    assert len(result["messages"]) == 2  # we do not add feedback on success


@pytest.mark.asyncio
async def test_try_again_workflow_custom_reward_threshold():
    """Test workflow with custom reward threshold."""

    async def fixed_reward_func(message) -> tuple[float, str]:
        return 0.7, "Consistent quality"

    model = FakeListChatModel(responses=["Response"])

    workflow_low = create_try_again_workflow(
        model=model,
        run_reward_func=fixed_reward_func,
        reward_score_for_pass=0.5,  # Lower threshold
        max_iterations=3,
    )

    result_low = await workflow_low.ainvoke({"messages": [HumanMessage(content="Test")]})
    assert result_low["turn"] == 1
    assert result_low["max_turns"] == 3

    workflow_high = create_try_again_workflow(
        model=model,
        run_reward_func=fixed_reward_func,
        reward_score_for_pass=0.9,  # Higher threshold
        max_iterations=2,
    )

    result_high = await workflow_high.ainvoke({"messages": [HumanMessage(content="Test")]})
    assert result_high["turn"] == 2
    assert result_high["max_turns"] == 2
    rewards = [message.additional_kwargs["reward_score"] for message in result_high["messages"] if
               isinstance(message, AIMessage)]
    assert all(reward == 0.7 for reward in rewards)
