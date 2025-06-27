from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
import pytest

from envbench_graphs.rebench_readonly.graph import create_rebench_readonly_workflow


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
