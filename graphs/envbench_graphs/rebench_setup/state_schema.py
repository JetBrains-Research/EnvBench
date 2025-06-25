from typing import Dict, List, TypedDict

from langchain_core.messages import BaseMessage


class RebenchSetupState(TypedDict):
    """State schema for the rebench-setup workflow."""
    
    # Messages
    messages: List[BaseMessage]
    
    # Repository info
    repo_name: str
    tools_kwargs: Dict
    
    # File processing
    files_tree: str
    file_list: List[str]
    file_contents: Dict[str, str]
    
    # Output
    setup_json: Dict
    setup_script: str 