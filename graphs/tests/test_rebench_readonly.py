from envbench_graphs.rebench_readonly.graph import create_rebench_readonly_workflow
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
import pytest


@pytest.mark.asyncio
async def test_rebench_readonly_workflow_compilation():
    """Test that the rebench-readonly workflow compiles correctly."""

    # Use a fake model for testing
    model = FakeListChatModel(responses=["Test response"])

    # Create the workflow
    workflow = create_rebench_readonly_workflow(model)

    # Test that it compiles without errors
    assert workflow is not None
    print("Workflow compiled successfully!")


@pytest.mark.asyncio
async def test_rebench_readonly_workflow_no_summarization():
    """Test the rebench-readonly workflow without summarization."""

    # Skip this test if no API key is available
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No OpenAI API key available")

    from langchain_openai import ChatOpenAI

    # Initialize the model
    model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=4000)

    # Create the workflow without summarization
    workflow = create_rebench_readonly_workflow(
        model=model, exploration_model=model, do_summarization=False, max_turns=3
    )

    # Test with a real repository
    initial_state = {
        "messages": [],
        "tools_kwargs": {"repository": "ansible/molecule"},
    }

    print("\n" + "=" * 50)
    print("REBENCH-READONLY WORKFLOW TEST (NO SUMMARIZATION)")
    print("=" * 50)
    print(f"Repository: {initial_state['tools_kwargs']['repository']}")

    # Run the workflow
    result = await workflow.ainvoke(initial_state, {"recursion_limit": 100})

    # Print the result
    print(f"\nMessages count: {len(result.get('messages', []))}")

    # Print messages using pretty_print
    messages = result.get("messages", [])
    if messages:
        print("\nMessages:")
        for i, msg in enumerate(messages, 1):
            print(f"\n--- Message {i} ---")
            print(msg.pretty_print())

    print("\n" + "=" * 50)
    print("✅ Workflow completed successfully!")


@pytest.mark.asyncio
async def test_rebench_readonly_workflow_with_summarization():
    """Test the rebench-readonly workflow with summarization."""

    # Skip this test if no API key is available
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No OpenAI API key available")

    from langchain_openai import ChatOpenAI

    # Initialize the model
    model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=4000)

    # Create the workflow with summarization
    workflow = create_rebench_readonly_workflow(
        model=model, exploration_model=model, do_summarization=True, max_turns=3
    )

    # Test with a real repository
    initial_state = {
        "messages": [],
        "tools_kwargs": {"repository": "ansible/molecule"},
    }

    print("\n" + "=" * 50)
    print("REBENCH-READONLY WORKFLOW TEST (WITH SUMMARIZATION)")
    print("=" * 50)
    print(f"Repository: {initial_state['tools_kwargs']['repository']}")

    # Run the workflow
    result = await workflow.ainvoke(initial_state, {"recursion_limit": 100})

    # Print the result
    print(f"\nMessages count: {len(result.get('messages', []))}")

    # Print messages using pretty_print
    messages = result.get("messages", [])
    if messages:
        print("\nMessages:")
        for i, msg in enumerate(messages, 1):
            print(f"\n--- Message {i} ---")
            print(msg.pretty_print())

    print("\n" + "=" * 50)
    print("✅ Workflow completed successfully!")


@pytest.mark.asyncio
async def test_rebench_readonly_workflow_skip_to_bash():
    """Test the rebench-readonly workflow skipping to bash script generation."""

    # Skip this test if no API key is available
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No OpenAI API key available")

    from langchain_openai import ChatOpenAI

    # Initialize the model
    model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=4000)

    # Create the workflow with entrypoint_node set to optional_summarization
    workflow = create_rebench_readonly_workflow(
        model=model,
        exploration_model=model,
        do_summarization=False,
        max_turns=3,
        entrypoint_node="optional_summarization",
    )

    # Test with existing messages
    initial_state = {
        "messages": [
            HumanMessage(content="I explored the repository and found it's a Python project with requirements.txt"),
            AIMessage(content="Great! I can see it's a Python project. Let me check the requirements.txt file."),
            HumanMessage(content="The requirements.txt contains: numpy, pandas, pytest"),
            AIMessage(content="Perfect! Now I understand the dependencies."),
        ],
        "tools_kwargs": {"repository": "ansible/molecule"},
    }

    print("\n" + "=" * 50)
    print("REBENCH-READONLY WORKFLOW TEST (SKIP TO BASH)")
    print("=" * 50)
    print(f"Repository: {initial_state['tools_kwargs']['repository']}")
    print(f"Initial messages: {len(initial_state['messages'])}")

    # Run the workflow
    result = await workflow.ainvoke(initial_state)

    # Print the result
    print(f"\nMessages count: {len(result.get('messages', []))}")

    # Print messages using pretty_print
    messages = result.get("messages", [])
    if messages:
        print("\nMessages:")
        for i, msg in enumerate(messages, 1):
            print(f"\n--- Message {i} ---")
            print(msg.pretty_print())

    print("\n" + "=" * 50)
    print("✅ Workflow completed successfully!")


@pytest.mark.asyncio
async def test_rebench_readonly_workflow_message_order_and_count():
    """Test the order and count of AI messages in a no summarization setup."""

    # Skip this test if no API key is available
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No OpenAI API key available")

    from langchain_openai import ChatOpenAI

    # Initialize the model
    model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=4000)

    # Create the workflow without summarization
    max_turns = 3
    workflow = create_rebench_readonly_workflow(
        model=model, exploration_model=model, do_summarization=False, max_turns=max_turns
    )

    # Test with a real repository
    initial_state = {
        "messages": [],
        "tools_kwargs": {"repository": "ansible/molecule"},
    }

    print("\n" + "=" * 50)
    print("REBENCH-READONLY WORKFLOW MESSAGE ORDER AND COUNT TEST")
    print("=" * 50)
    print(f"Repository: {initial_state['tools_kwargs']['repository']}")
    print(f"Max turns: {max_turns}")

    # Run the workflow
    result = await workflow.ainvoke(initial_state, {"recursion_limit": 100})

    # Get all messages
    messages = result.get("messages", [])
    print(f"\nTotal messages: {len(messages)}")

    # Extract different message types
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]

    print(f"AI messages count: {len(ai_messages)}")
    print(f"Human messages count: {len(human_messages)}")
    print(f"Tool messages count: {len(tool_messages)}")

    # Check that we have <= max_turns AI messages with tool calls (excluding the final bash script generation)
    ai_messages_with_tool_calls = [msg for msg in ai_messages if hasattr(msg, "tool_calls") and msg.tool_calls]
    print(f"AI messages with tool calls: {len(ai_messages_with_tool_calls)}")

    # The final AI message (bash script generation) should not have tool calls
    final_ai_message = ai_messages[-1] if ai_messages else None
    final_has_tool_calls = (
        (hasattr(final_ai_message, "tool_calls") and final_ai_message.tool_calls) if final_ai_message else False
    )

    # Count AI messages with tool calls (excluding the final one)
    exploration_ai_messages_with_tool_calls = (
        ai_messages_with_tool_calls[:-1] if final_has_tool_calls else ai_messages_with_tool_calls
    )
    exploration_count = len(exploration_ai_messages_with_tool_calls)

    print(f"Exploration AI messages with tool calls: {exploration_count}")

    # Check that we have <= max_turns exploration AI messages with tool calls
    assert exploration_count <= max_turns, (
        f"Expected <= {max_turns} exploration AI messages with tool calls, got {exploration_count}"
    )

    # Check that at most one AI message has the stop content
    stop_messages = [msg for msg in ai_messages if msg.content == "Sorry, need more steps to process this request."]
    stop_count = len(stop_messages)
    print(f"Stop messages count: {stop_count}")

    assert stop_count <= 1, f"Expected at most 1 stop message, got {stop_count}"

    # Print all message details for debugging
    print("\nAll Messages Details:")
    for i, msg in enumerate(messages, 1):
        msg_type = type(msg).__name__
        has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
        is_stop = msg.content == "Sorry, need more steps to process this request." if hasattr(msg, "content") else False

        print(f"  {i}. {msg_type} - Tool calls: {has_tool_calls}, Stop message: {is_stop}")

        if has_tool_calls:
            print(f"     Tool calls: {len(msg.tool_calls)}")
            for j, tool_call in enumerate(msg.tool_calls):
                print(f"       {j + 1}. {tool_call.get('name', 'unknown')}: {tool_call.get('args', {})}")

        if hasattr(msg, "content") and msg.content:
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"     Content preview: {content_preview}")

        if hasattr(msg, "tool_call_id"):
            print(f"     Tool call ID: {msg.tool_call_id}")

    print("\n" + "=" * 50)
    print("✅ Message order and count test passed!")
    print(f"   - Exploration AI messages with tool calls: {exploration_count} (<= {max_turns})")
    print(f"   - Stop messages: {stop_count} (<= 1)")
    print(f"   - Total AI messages: {len(ai_messages)}")
    print(f"   - Total Human messages: {len(human_messages)}")
    print(f"   - Total Tool messages: {len(tool_messages)}")
