import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Set

from datasets import load_dataset  # type: ignore[import-untyped]
from env_setup_utils.markdown import extract_headings_with_keywords


class EnvSetupInstructionProvider(ABC):
    @abstractmethod
    def __call__(self, repository: str, revision: str) -> Optional[str]: ...


class EmptyEnvSetupInstructionProvider(EnvSetupInstructionProvider):
    def __call__(self, repository: str, revision: str) -> Optional[str]:
        return None


class READMEEnvSetupInstructionProvider(EnvSetupInstructionProvider, ABC):
    def __init__(self, dataset_name: str, language: str):
        self.dataset_name = dataset_name
        self.language = language
        self.dataset = load_dataset(dataset_name, "readmes", split=language)

    def _get_readme(self, repository: str, revision: str) -> str:
        datapoints = self.dataset.filter(lambda dp: dp["repository"] == repository and dp["revision"] == revision)
        if len(datapoints) == 0:  # type: ignore[reportArgumentType]
            raise RuntimeError(f"No readme available for {repository}@{revision}")
        if len(datapoints) > 1:  # type: ignore[reportArgumentType]
            logging.warning(f"Multiple readmes available for {repository}@{revision}. Will choose the first entry.")
        datapoint = datapoints[0]  # type: ignore[reportIndexIssue]
        return datapoint["contents"]  # type: ignore[reportReturnType]

    @abstractmethod
    def _filter_readme(self, readme: str) -> str: ...

    def __call__(self, repository: str, revision: str) -> Optional[str]:
        try:
            readme = self._get_readme(repository=repository, revision=revision)
            return self._filter_readme(readme)
        except Exception as e:
            logging.exception(e)
            return None


class SimpleREADMEEnvSetupInstructionProvider(READMEEnvSetupInstructionProvider):
    build_heading_keywords: Set[str] = {"build", "install", "start"}

    def _filter_readme(self, readme: str) -> str:
        build_headings = extract_headings_with_keywords(readme, keywords=self.build_heading_keywords)
        resulting_readme: List[str] = []

        for heading in build_headings:
            cur_contents = f"{'#' * heading['heading_level']} {heading['heading']}\n\n{heading['contents']}"
            resulting_readme.append(cur_contents)
        return "\n\n".join(resulting_readme)
