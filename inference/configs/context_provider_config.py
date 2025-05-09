from pydantic import Extra

from inference.src.context_providers.build_instructions import EnvSetupInstructionProvider

from .instantiatable_config import InstantiatableConfig


class EnvSetupInstructionProviderConfig(InstantiatableConfig[EnvSetupInstructionProvider], extra=Extra.allow): ...
