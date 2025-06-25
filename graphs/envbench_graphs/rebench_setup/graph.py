import json
import re
from typing import Awaitable, Callable, Dict, Optional

import aiohttp
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import END, StateGraph

from .prompts import GENERATE_SETUP_PROMPT, LIST_FILES_PROMPT
from .state_schema import RebenchSetupState


async def default_execute_bash_command(bash_command: str, repository: str) -> Dict[str, str]:
    """Default implementation of execute_bash_command using the provided endpoint."""
    async with aiohttp.ClientSession() as session:
        payload = {
            "bash_command": bash_command,
            "repository": repository
        }
        async with session.post(
            "https://envbench-explorer.wlko.me/execute",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            return {
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "exit_code": result.get("exit_code", 1),
                "success": result.get("success", False)
            }


def filter_files(files: list[str]) -> list[str]:
    """Filter out common non-relevant files from the file list."""
    skip_patterns = [
        '.git/', '.gitignore', '.gitattributes',
        '__pycache__/', '.pyc', '.pyo',
        '.DS_Store', 'Thumbs.db',
        '*.log', '*.tmp', '*.swp',
        'node_modules/', '.npm/',
        '.pytest_cache/', '.coverage',
        'build/', 'dist/', '*.egg-info/',
        '.mypy_cache/', '.ruff_cache/',
        '.vscode/', '.idea/',
    ]
    
    filtered_files = []
    for file_path in files:
        clean_path = file_path.lstrip('./')
        if not clean_path:
            continue
            
        should_skip = any(pattern in clean_path for pattern in skip_patterns)
        if not should_skip:
            filtered_files.append(clean_path)
    
    return filtered_files


def parse_file_contents(output: str) -> Dict[str, str]:
    """Parse file contents from bash command output."""
    file_contents = {}
    file_sections = re.split(r'===START_FILE:(.*?)===', output)
    
    for i in range(1, len(file_sections), 2):
        if i + 1 < len(file_sections):
            file_path = file_sections[i].strip()
            content = file_sections[i + 1].split("===END_FILE:")[0].strip()
            
            if not content.startswith("No such file:"):
                file_contents[file_path] = content
    
    return file_contents


def extract_json_array(content: str) -> Optional[list[str]]:
    """Extract JSON array from model response."""
    try:
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            file_list = json.loads(json_match.group())
            if isinstance(file_list, list) and all(isinstance(x, str) for x in file_list):
                return file_list
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def extract_setup_artifacts(content: str) -> tuple[Dict, str]:
    """Extract JSON and bash script from model response."""
    setup_json = {}
    setup_script = ""
    
    # Extract JSON from markdown code block
    try:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            setup_json = json.loads(json_match.group(1))
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract bash script from markdown code block
    bash_match = re.search(r'```bash\s*(.*?)\s*```', content, re.DOTALL)
    if bash_match:
        setup_script = bash_match.group(1).strip()
    
    return setup_json, setup_script


def create_rebench_setup_workflow(
    model: BaseChatModel,
    file_selection_model: Optional[BaseChatModel] = None,
    execute_bash_command: Optional[Callable[[str, str], Awaitable[Dict[str, str]]]] = None,
) -> CompiledGraph:
    """Create a compiled workflow graph for rebench-setup operations."""
    
    if execute_bash_command is None:
        execute_bash_command = default_execute_bash_command
    
    # Use the same model for file selection if none provided
    if file_selection_model is None:
        file_selection_model = model

    def initialize_state(state: RebenchSetupState) -> RebenchSetupState:
        """Initialize the state with default values."""
        return {
            "messages": state.get("messages", []),
            "repo_name": state.get("repo_name", "unknown"),
            "tools_kwargs": state.get("tools_kwargs", {}),
            "files_tree": state.get("files_tree", ""),
            "file_list": state.get("file_list", []),
            "file_contents": state.get("file_contents", {}),
            "setup_json": state.get("setup_json", {}),
            "setup_script": state.get("setup_script", ""),
        }

    async def fetch_and_analyze_files(state: RebenchSetupState) -> RebenchSetupState:
        """Fetch files tree, prompt model for file list, and fetch file contents."""
        repository = state["tools_kwargs"].get("repository", state["repo_name"]).replace('/', '__')
        
        # Step 1: Fetch files tree
        bash_command = 'find . -type f -name "*" | sort'
        result = await execute_bash_command(bash_command, repository)
        
        if not result["success"]:
            return state
        
        files = result["stdout"].strip().split('\n')
        filtered_files = filter_files(files)
        files_tree = '\n'.join(filtered_files)
        
        # Step 2: Prompt model for file list (using file_selection_model)
        repo_name = state["tools_kwargs"].get("repository", state["repo_name"]).replace('/', '__')
        message = HumanMessage(content=LIST_FILES_PROMPT.format(
            repo_name=repo_name,
            files_tree=files_tree
        ))
        
        response = await file_selection_model.ainvoke([message])
        content = str(response.content)
        file_list = extract_json_array(content)
        
        if not file_list:
            return state | {"files_tree": files_tree}
        
        # Step 3: Fetch file contents
        file_commands = []
        for file_path in file_list:
            file_commands.append(f'echo "===START_FILE:{file_path}==="')
            file_commands.append(
                f'if [ -f "{file_path}" ]; then cat "{file_path}"; '
                f'else echo "No such file: {file_path}"; fi'
            )
            file_commands.append(f'echo "===END_FILE:{file_path}==="')
        
        bash_command = " && ".join(file_commands)
        result = await execute_bash_command(bash_command, repository)
        
        if not result["success"]:
            return state | {"files_tree": files_tree, "file_list": file_list}
        
        file_contents = parse_file_contents(result["stdout"])
        
        return state | {
            "files_tree": files_tree,
            "file_list": file_list,
            "file_contents": file_contents,
        }

    async def generate_setup(state: RebenchSetupState) -> RebenchSetupState:
        """Generate setup instructions from file contents."""
        if not state["file_contents"]:
            return state
        
        # Prepare file contents text
        file_contents_text = ""
        for file_path, content in state["file_contents"].items():
            file_contents_text += f"\n--- {file_path} ---\n{content}\n"

        if len(file_contents_text) > 100000:
            file_contents_text = file_contents_text[:100000]

        repo_name = state["tools_kwargs"].get("repository", state["repo_name"]).replace('/', '__')
        full_prompt = GENERATE_SETUP_PROMPT.format(
            repo_name=repo_name,
            rendered=file_contents_text
        )
        message = HumanMessage(content=full_prompt)
        
        response = await model.ainvoke([message])
        content = str(response.content)
        setup_json, setup_script = extract_setup_artifacts(content)
        
        return state | {
            "messages": [message, response],
            "setup_json": setup_json,
            "setup_script": setup_script,
        }

    # Create the graph
    workflow = StateGraph(RebenchSetupState)

    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("fetch_and_analyze_files", fetch_and_analyze_files)
    workflow.add_node("generate_setup", generate_setup)

    # Add edges
    workflow.add_edge("initialize", "fetch_and_analyze_files")
    workflow.add_edge("fetch_and_analyze_files", "generate_setup")
    workflow.add_edge("generate_setup", END)

    # Set entry point
    workflow.set_entry_point("initialize")

    return workflow.compile() 