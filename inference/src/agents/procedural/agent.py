from typing import List, Optional, TypedDict, Literal
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage
from langgraph.graph.graph import CompiledGraph

from .commands import PYTHON_CONTEXT_COMMANDS, JVM_CONTEXT_COMMANDS
from .prompts import get_python_setup_prompt, get_jvm_setup_prompt
from ...async_bash_executor import CommandExecutionResult
from ...context_providers.build_instructions import EnvSetupInstructionProvider
from ...toolkits.base import BaseEnvSetupToolkit
from ...utils import message_to_info
from ..base import BaseEnvSetupAgent


class EnvSetupProceduralState(TypedDict):
    build_instructions: str
    context: str
    script: Optional[str]
    messages: List[BaseMessage]


class EnvSetupProceduralUpdate(TypedDict):
    timestamp: datetime
    node: str
    messages: List[BaseMessage]


class EnvSetupProceduralTrajectoryEntry(TypedDict):
    timestamp: datetime
    node: str
    messages: List[dict]


class EnvSetupProceduralAgent(BaseEnvSetupAgent[EnvSetupProceduralState,
                                               EnvSetupProceduralUpdate,
                                               EnvSetupProceduralTrajectoryEntry]):
    def __init__(
        self,
        model: BaseChatModel,
        toolkit: BaseEnvSetupToolkit,
        instruction_provider: EnvSetupInstructionProvider,
        language: Literal["python", "jvm"],
        max_iterations: Optional[int] = None,
    ):
        self.toolkit = toolkit
        self.model = model
        self.instruction_provider = instruction_provider
        self.language = language
        self._max_iterations = max_iterations
        self._resulting_commands: List[CommandExecutionResult] = []

    async def collect_context(self, state: EnvSetupProceduralState) -> dict:
        """Node that collects context by running predefined commands."""
        commands = PYTHON_CONTEXT_COMMANDS if self.language == "python" else JVM_CONTEXT_COMMANDS
        
        results = []
        for cmd in commands:
            result, exit_code = await self.toolkit.bash_executor.execute_bash_command(cmd)
            results.append(f"Command: {cmd}\nOutput: {result}\nExit Code: {exit_code}")
        
        context = "\n".join(results)
        res = state.copy()
        res["context"] = context
        return res

    async def generate_script(self, state: EnvSetupProceduralState) -> dict:
        """Node that generates the script using the LLM."""
        prompt_func = get_python_setup_prompt if self.language == "python" else get_jvm_setup_prompt
        prompt = prompt_func(state)
        print(prompt)
        response = await self.model.ainvoke(prompt)
        script = response.content
        
        # Extract the script from the bash code block
        if "```bash" in script and "```" in script:
            bash_script = script.split("```bash", 1)[1].split("```", 1)[0].strip()
            # Store the entire script as a single command
            self._resulting_commands = [CommandExecutionResult(command=bash_script, exit_code=None)]
        else:
            self._resulting_commands = []

        res = state.copy()
        res["messages"] = [response]
        res["script"] = script
        return res

    @property
    def max_iterations(self) -> Optional[int]:
        return 5  # We only need one iteration for the procedural flow

    @property
    def commands_history(self) -> List[CommandExecutionResult]:
        return self._resulting_commands

    def get_agent(self) -> CompiledGraph:
        workflow = StateGraph(EnvSetupProceduralState)
        
        # Add nodes
        workflow.add_node("context_collector", self.collect_context)
        workflow.add_node("script_generator", self.generate_script)
        
        # Add edges
        workflow.set_entry_point("context_collector")
        workflow.add_edge("context_collector", "script_generator")
        workflow.add_edge("script_generator", END)

        return workflow.compile()

    def construct_initial_state(self, repository: str, revision: str, *args, **kwargs) -> EnvSetupProceduralState:
        instructions = self.instruction_provider(repository=repository, revision=revision)
        if instructions is None:
            instructions = ""  # Provide a default empty string if None
        return {
            "build_instructions": instructions,
            "context": "",
            "script": None
        }

    @staticmethod
    def process_update_for_trajectory(
        update: EnvSetupProceduralUpdate, *args, **kwargs
    ) -> EnvSetupProceduralTrajectoryEntry:
        node = "unknown"
        messages = []
        if "context_collector" in update:
            node = "context_collector"
            messages = update["context_collector"].get("messages", [])
        elif "script_generator" in update:
            node = "script_generator"
            messages = update["script_generator"].get("messages", [])
        return {
            "timestamp": update["timestamp"],
            "node": node,
            "messages": [message_to_info(msg) for msg in messages],
        }
