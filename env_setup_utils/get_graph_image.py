#!/usr/bin/env python3
"""
Script to generate visualization images for all LangGraph graphs in graphs/envbench_graphs.
"""

import importlib
from pathlib import Path
import sys
from typing import Any, Dict, List

from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

# Add the graphs directory to the Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
graphs_dir = project_root / "graphs"
sys.path.insert(0, str(graphs_dir))


class DummyTool(BaseTool):
    """Dummy tool for testing purposes."""

    name: str = "dummy_tool"
    description: str = "A dummy tool for testing"

    def _run(self, input: str) -> str:
        return f"Dummy response for: {input}"

    async def _arun(self, input: str) -> str:
        return f"Dummy async response for: {input}"


@tool
def dummy_submit_shell_script_tool(script: str) -> str:
    """Submit a shell script."""
    return f"Submitted script: {script[:50]}..."


async def dummy_shellcheck_func(script: str) -> str:
    """Dummy shellcheck function."""
    return f"Shellcheck result for script: {script[:50]}..."


async def dummy_execute_bash_command(command: str, repository: str) -> Dict[str, Any]:
    """Dummy bash execution function."""
    return {"exit_code": 0, "stdout": f"Output for command: {command}", "stderr": "", "success": True}


def get_graph_modules() -> List[str]:
    """Get all graph module names from envbench_graphs directory."""
    envbench_graphs_dir = graphs_dir / "envbench_graphs"
    modules = []

    for item in envbench_graphs_dir.iterdir():
        if item.is_dir() and not item.name.startswith("__") and not item.name == "data":
            graph_file = item / "graph.py"
            if graph_file.exists():
                modules.append(item.name)

    return modules


def create_dummy_model():
    """Create a dummy model for testing."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key="dummy-key",
        base_url="http://localhost:8000",  # This won't be called for graph structure
    )


def create_graph_for_module(module_name: str):
    """Create a graph for a specific module."""
    try:
        # Import the module
        module = importlib.import_module(f"envbench_graphs.{module_name}.graph")

        # Get all create functions
        create_functions = [
            name for name in dir(module) if name.startswith("create_") and callable(getattr(module, name))
        ]

        # Filter to only locally defined functions (not imported)
        local_create_functions = []
        for func_name in create_functions:
            func = getattr(module, func_name)
            # Check if the function is defined in this module (not imported)
            if func.__module__ == module.__name__:
                local_create_functions.append(func_name)

        # Use local functions if available, otherwise fall back to all functions
        if local_create_functions:
            create_functions = local_create_functions

        if not create_functions:
            print(f"No create function found in {module_name}")
            return None

        create_func = getattr(module, create_functions[0])

        # Create dummy parameters based on module
        model = create_dummy_model()

        if module_name == "readonly":
            tools = [DummyTool()]
            graph = create_func(
                model=model, tools=tools, submit_shell_script_tool=dummy_submit_shell_script_tool, max_iterations=3
            )

        elif module_name == "multi_attempt":
            graph = create_func(model=model, max_iterations=2)

        elif module_name == "shellcheck":
            graph = create_func(model=model, run_shellcheck_func=dummy_shellcheck_func, max_iterations=2)

        elif module_name == "rebench_readonly":
            graph = create_func(
                model=model,
                exploration_model=model,
                do_summarization=False,
                max_turns=5,
                execute_bash_command=dummy_execute_bash_command,
            )

        elif module_name == "rebench_setup":
            graph = create_func(
                model=model, file_selection_model=model, execute_bash_command=dummy_execute_bash_command
            )

        else:
            # Try with just model parameter as fallback
            graph = create_func(model=model)

        return graph

    except Exception as e:
        print(f"Error creating graph for {module_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def save_graph_image(graph, module_name: str, output_dir: Path):
    """Save graph visualization to PNG file using advanced mermaid styling."""
    try:
        # Get the graph structure
        graph_def = graph.get_graph()

        # Generate styled PNG image using mermaid
        try:
            png_file = output_dir / f"{module_name}_graph.png"

            png_data = graph_def.draw_mermaid_png(
                curve_style=CurveStyle.LINEAR,
                node_colors=NodeStyles(
                    first="#ffdfba",  # Light orange for start nodes
                    last="#baffc9",  # Light green for end nodes
                    default="#fad7de",  # Light pink for default nodes
                ),
                wrap_label_n_words=9,
                output_file_path=None,  # Return bytes instead of saving to file
                draw_method=MermaidDrawMethod.PYPPETEER,
                background_color="white",
                padding=10,
            )

            with open(png_file, "wb") as f:
                f.write(png_data)
            print(f"✓ Saved styled PNG diagram for {module_name} to {png_file}")
            return True

        except Exception as e:
            print(f"✗ Could not generate styled PNG for {module_name}: {e}")
            return False

    except Exception as e:
        print(f"✗ Error saving graph image for {module_name}: {e}")
        return False


def main():
    """Main function to generate all graph images."""
    print("Generating styled PNG images for all LangGraph graphs...")

    # Create output directory
    output_dir = project_root / "graph_images"
    output_dir.mkdir(exist_ok=True)

    # Get all graph modules
    modules = get_graph_modules()
    print(f"Found graph modules: {modules}")

    successful = 0
    total = len(modules)

    for module_name in modules:
        print(f"\n📊 Processing {module_name}...")

        # Create the graph
        graph = create_graph_for_module(module_name)

        if graph is not None:
            # Save PNG visualization
            if save_graph_image(graph, module_name, output_dir):
                successful += 1
        else:
            print(f"✗ Failed to create graph for {module_name}")

    print("\n" + "=" * 50)
    print(f"📈 Summary: Successfully processed {successful}/{total} graphs")
    print(f"📁 Output directory: {output_dir}")

    # List generated PNG files
    png_files = list(output_dir.glob("*.png"))
    if png_files:
        print("\n🖼️  Generated PNG files:")
        for file in sorted(png_files):
            print(f"  {file.name}")
    else:
        print("\n⚠️  No PNG files were generated")


if __name__ == "__main__":
    main()
