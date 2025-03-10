import shlex
from typing import Annotated, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from ..utils.installamatic import NON_NL, get_headings, get_headings_rst
from .base import BaseEnvSetupToolkit


class InstallamaticToolkit(BaseEnvSetupToolkit):
    async def get_directory_contents(
        self, directory: str = Field(description="The path to the directory to be inspected.")
    ):
        """
        Retrieve the contents of a given directory, including any files and subdirectories it contains.
        """
        if directory == "." or directory == "/":
            dir_contents, exit_code = await self._execute_bash_command("ls")
            assert exit_code == 0, f"Couldn't obtain contents of {directory}. Response: {dir_contents}"
            return dir_contents

        dir_contents, exit_code = await self._execute_bash_command(f"ls {shlex.quote(directory)}")
        assert exit_code == 0, f"Couldn't obtain contents of {directory}. Response: {dir_contents}"
        return dir_contents

    async def get_file_contents(
        self,
        state: Annotated[dict, InjectedState],
        file: str = Field(description="The path to the file to be inspected."),
    ):
        """
        Retrieve the contents of a given file, or a list of headings if the file has .md or .rst extension.
        """
        stage = state.get("stage")
        if stage == "build":
            documentation = state.get("documentation", {})
            if file not in documentation:
                documentation_str = "\n".join(f"- {file}" for file in state.get("documentation", []))
                return f"The file path {file} is not present among gathered documentation files that you're allowed to access. Please, only access one of:\n{documentation_str}"

        file_contents, exit_code = await self._execute_bash_command(f"cat {shlex.quote(file)}")

        assert exit_code == 0, f"Couldn't obtain contents of {file}. Response: {file_contents}"

        headings = get_headings_rst(file_contents) if file.lower().endswith(".rst") else get_headings(file_contents)

        if not any(non_nl_pattern in file for non_nl_pattern in NON_NL) and headings is not None:
            headings_str = "\n - ".join([h[0] for h in headings])
            function_response = f"\nhere are the section headers of the file: \n - {headings_str}"
        else:
            function_response = file_contents + "\n" + f"here are the contents of file {file}"

        return function_response

    async def inspect_header(
        self,
        file: str = Field(description="The path to the file to be inspected."),
        heading: str = Field(description="The name of the section header to inspect."),
    ):
        """
        Retrieve the contents of a given heading in a file
        """
        file_contents, exit_code = await self._execute_bash_command(f"cat {shlex.quote(file)}")

        assert exit_code == 0, f"Couldn't obtain contents of {file}. Response: {file_contents}"

        headings = get_headings_rst(file_contents) if file.lower().endswith(".rst") else get_headings(file_contents)

        if not any(non_nl_pattern in file for non_nl_pattern in NON_NL) and headings is not None:
            for hdg in headings:
                if hdg[0] == heading:
                    return f"here are the contents of {file}'s '{heading}' section:\n\n{hdg[1]}\n"
            return f"header '{heading}' can not be found in file {file}!"

        return f"Unable to retrieve headings for {file}. Please, refer to the full contents."

    async def check_presence(
        self,
        file: str = Field(description="The path to the file."),
    ):
        """
        Confirm whether the given file exists in the project.
        Use this to confirm any assumptions you make, to prevent hallucinations.
        """
        output, exit_code = await self._execute_bash_command(f"test -f {shlex.quote(file)}")
        if exit_code == 0:
            return f"{file} exists."
        return f"{file} does NOT exist."

    async def submit_shell_script(
        self,
        script: str = Field(description="The contents of the script."),
    ):
        """
        Given your current knowledge of the repo, provide a shell script to this
        function that sets up the repo.
        """
        return "Thank you! Shell script submitted."

    async def submit_documentation(
        self,
        file: str = Field(description="Path to a file that contains documentation relative to the root directory."),
    ):
        """
        Submit and record the path to a file containing documentation.
        """
        return "Thank you! File submitted."

    async def finished_search(
        self,
    ):
        """
        Signal that you have found and submitted all documentation in the repo.
        """
        return "Thank you! Please, proceed to generate a shell script."

    async def submit_summary(
        self,
        summary: str = Field(description="A summary of the information you have gathered in the previous step."),
    ):
        """
        Submit a summary of the information you have gathered.
        """
        return "Thank you!"

    def get_tools(self, stage: Optional[str] = None, *args, **kwargs) -> List[BaseTool]:
        if stage is None:
            return [
                StructuredTool.from_function(coroutine=self.get_directory_contents),
                StructuredTool.from_function(coroutine=self.get_file_contents),
                StructuredTool.from_function(coroutine=self.inspect_header),
                StructuredTool.from_function(coroutine=self.check_presence),
                StructuredTool.from_function(coroutine=self.submit_shell_script),
                StructuredTool.from_function(coroutine=self.submit_documentation),
                StructuredTool.from_function(coroutine=self.finished_search),
                StructuredTool.from_function(coroutine=self.submit_summary),
            ]

        if stage == "search":
            return [
                StructuredTool.from_function(coroutine=self.get_directory_contents),
                StructuredTool.from_function(coroutine=self.get_file_contents),
                StructuredTool.from_function(coroutine=self.inspect_header),
                StructuredTool.from_function(coroutine=self.check_presence),
                StructuredTool.from_function(coroutine=self.submit_documentation),
                StructuredTool.from_function(coroutine=self.finished_search),
            ]

        if stage == "build":
            return [
                StructuredTool.from_function(coroutine=self.get_directory_contents),
                StructuredTool.from_function(coroutine=self.get_file_contents),
                StructuredTool.from_function(coroutine=self.inspect_header),
                StructuredTool.from_function(coroutine=self.check_presence),
                StructuredTool.from_function(coroutine=self.submit_summary),
            ]

        if stage == "submit_shell_script":
            return [
                StructuredTool.from_function(coroutine=self.submit_shell_script),
            ]

        raise ValueError(f"Unknown stage {stage}. Currently supported are: search, build, None (to get all tools).")
