from envbench_graphs.rebench_setup.graph import create_rebench_setup_workflow
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
import pytest


@pytest.mark.asyncio
async def test_rebench_setup_workflow_compilation():
    """Test that the rebench-setup workflow compiles correctly."""

    # Use a fake model for testing
    model = FakeListChatModel(responses=["Test response"])

    # Create the workflow
    workflow = create_rebench_setup_workflow(model)

    # Test that it compiles without errors
    assert workflow is not None
    print("Workflow compiled successfully!")


@pytest.mark.asyncio
async def test_rebench_setup_workflow():
    """Test the rebench-setup workflow with real OpenAI API and endpoint."""

    # Skip this test if no API key is available
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No OpenAI API key available")

    from langchain_openai import ChatOpenAI

    # Initialize the model
    model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=4000)

    # Create the workflow
    workflow = create_rebench_setup_workflow(model)

    # Test with a real repository
    initial_state = {
        "messages": [],
        "tools_kwargs": {"repository": "ansible/molecule"},
    }

    # Run the workflow
    result = await workflow.ainvoke(initial_state)

    # Verify the result has the expected structure
    assert "setup_json" in result
    assert "setup_script" in result
    assert "file_contents" in result

    # Print the result for inspection
    print("\n" + "=" * 50)
    print("REBENCH-SETUP WORKFLOW TEST RESULTS")
    print("=" * 50)
    print(f"Repository: {initial_state['tools_kwargs']['repository']}")

    # Check files tree was generated
    files_tree = result.get("files_tree", "")
    print(f"\nFiles tree length: {len(files_tree)}")
    print("First 10 files:")
    for i, file_path in enumerate(files_tree.split("\n")[:10]):
        print(f"  {i + 1}. {file_path}")

    # Check file list was generated
    file_list = result.get("file_list", [])
    print(f"\nSelected files ({len(file_list)}):")
    for i, file_path in enumerate(file_list):
        print(f"  {i + 1}. {file_path}")

    # Check file contents were fetched
    file_contents = result.get("file_contents", {})
    print(f"\nFile contents fetched ({len(file_contents)} files):")
    for file_path, content in file_contents.items():
        print(f"  {file_path}: {len(content)} characters")

    # Check setup JSON was generated
    setup_json = result.get("setup_json", {})
    print(f"\nSetup JSON generated: {bool(setup_json)}")
    if setup_json:
        print("Setup JSON fields:")
        for key, value in setup_json.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")

    # Check bash script was generated
    setup_script = result.get("setup_script", "")
    print(f"\nBash script generated: {bool(setup_script)}")
    if setup_script:
        print(f"Bash script length: {len(setup_script)} characters")
        print("First 200 characters of bash script:")
        print(setup_script[:200] + "..." if len(setup_script) > 200 else setup_script)

    print("\n" + "=" * 50)

    # Assertions to verify the workflow produced output
    assert result is not None, "Workflow should return a result"
    assert result.get("files_tree", ""), "Files tree should be generated"
    assert result.get("file_list", []), "File list should be generated"
    assert result.get("file_contents", {}), "File contents should be fetched"
    assert result.get("setup_json", {}), "Setup JSON should be generated"
    assert result.get("setup_script", ""), "Bash script should be generated"

    # Verify messages from generate_setup stage
    messages = result.get("messages", [])
    assert len(messages) == 2, f"Should have exactly 2 messages from generate_setup stage, got {len(messages)}"
    assert isinstance(messages[0], HumanMessage), "First message should be HumanMessage"
    assert hasattr(messages[1], "content"), "Second message should be AI response"

    print("✅ All assertions passed! Workflow completed successfully.")
