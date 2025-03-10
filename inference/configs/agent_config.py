from enum import Enum
from typing import Optional, Union

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Extra, validator

from configs.context_provider_config import EnvSetupInstructionProviderConfig
from configs.instantiatable_config import InstantiatableConfig
from configs.toolkit_config import EnvSetupToolkit
from src.agents.base import BaseEnvSetupAgent
from src.agents.installamatic.agent import InstallamaticAgent
from src.agents.jvm.agent import EnvSetupJVMAgent
from src.agents.python.agent import EnvSetupPythonAgent
from src.toolkits.base import BaseEnvSetupToolkit

from src.agents.procedural.agent import EnvSetupProceduralAgent


class ModelConfig(InstantiatableConfig[BaseChatModel], extra=Extra.allow): ...


class EnvSetupAgentType(Enum):
    python = "python"
    jvm = "jvm"
    procedural_python = "procedural-python"
    procedural_jvm = "procedural-jvm"
    installamatic = "installamatic"


class EnvSetupAgentConfig(BaseModel):
    agent_type: EnvSetupAgentType
    """Which agent to instantiate."""
    toolkit: EnvSetupToolkit
    """Defines tools available for the agent."""
    model: ModelConfig
    """Defines which LLM is used as the agent's backbone."""
    instruction_provider: EnvSetupInstructionProviderConfig
    """Defines logic of obtaining build instructions for given datapoint (repository@revision)."""
    max_iterations: int
    """Defines maximum allowed number of iterations."""
    language: Optional[str] = None

    @validator("toolkit")
    def validate_toolkit(cls, toolkit: Union[str, EnvSetupToolkit]) -> EnvSetupToolkit:
        if isinstance(toolkit, str):
            return EnvSetupToolkit(toolkit)
        return toolkit

    def instantiate(self, toolkit: BaseEnvSetupToolkit) -> BaseEnvSetupAgent:
        model = self.model.instantiate()
        instruction_provider = self.instruction_provider.instantiate()

        if self.agent_type == EnvSetupAgentType.python or self.agent_type == EnvSetupAgentType.python.value:
            return EnvSetupPythonAgent(
                toolkit=toolkit,
                model=model,
                instruction_provider=instruction_provider,
                max_iterations=self.max_iterations,
            )

        if self.agent_type == EnvSetupAgentType.jvm or self.agent_type == EnvSetupAgentType.jvm.value:
            return EnvSetupJVMAgent(
                toolkit=toolkit,
                model=model,
                instruction_provider=instruction_provider,
                max_iterations=self.max_iterations,
            )

        if (
            self.agent_type == EnvSetupAgentType.installamatic
            or self.agent_type == EnvSetupAgentType.installamatic.value
        ):
            return InstallamaticAgent(
                toolkit=toolkit, model=model, max_iterations=self.max_iterations, language=self.language
            )

        if self.agent_type == EnvSetupAgentType.procedural_python or self.agent_type == EnvSetupAgentType.procedural_python.value:
            return EnvSetupProceduralAgent(
                toolkit=toolkit,
                model=model,
                instruction_provider=instruction_provider,
                max_iterations=self.max_iterations,
                language="python",
            )

        if self.agent_type == EnvSetupAgentType.procedural_jvm or self.agent_type == EnvSetupAgentType.procedural_jvm.value:
            return EnvSetupProceduralAgent(
                toolkit=toolkit,
                model=model,
                instruction_provider=instruction_provider,
                max_iterations=self.max_iterations,
                language="jvm",
            )

        raise ValueError(
            f"Expected agent_type to be `EnvSetupAgentType`, but got {self.agent_type} ({type(self.agent_type)})."
        )
