import json
import logging
from typing import List

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field

from inference.src.toolkits.base import BaseEnvSetupToolkit

logger = logging.getLogger(__name__)


class ShellcheckToolkit(BaseEnvSetupToolkit):
    def initial_commands(self):
        # Simple shellcheck installation - x86_64 should have better compatibility
        return super().initial_commands() + ["apt-get update && apt-get install -y shellcheck"]

    async def _get_readable_shellcheck_outputs(self, script: str) -> str:
        """Executes shellcheck on a given script and returns a readable string."""
        try:
            shellcheck_results_str = await self._run_shellcheck(script)
        except Exception as e:
            return f"Error running shellcheck: {repr(e)}"

        # Handle the case where AsyncBashExecutor wraps the output due to non-zero exit code
        # OR when it formats output with stdout/stderr even on success
        if (
            shellcheck_results_str.startswith("ERROR: Could not execute given command.")
            or "\nstdout:\n" in shellcheck_results_str
            or shellcheck_results_str.startswith("stdout:\n")
        ):
            # Parse the wrapped output to get just the stdout
            lines = shellcheck_results_str.split("\n")
            stdout_start = False
            stdout_lines = []
            for line in lines:
                if line == "stdout:":
                    stdout_start = True
                    continue
                elif line == "stderr:":
                    break
                elif stdout_start:
                    stdout_lines.append(line)
            shellcheck_results_str = "\n".join(stdout_lines)

        try:
            shellcheck_results = json.loads(shellcheck_results_str)
        except json.JSONDecodeError:
            logging.warning(f"Error parsing shellcheck results: {repr(shellcheck_results_str)}, returning raw results.")
            return shellcheck_results_str

        if shellcheck_results is None:
            return "Unable to obtain shellcheck results. Please, check the submitted script and try again."

        if len(shellcheck_results) == 0:
            return "No issues found."

        result_str = []
        result_str.append(
            f"Got {len(shellcheck_results)} issues reported by shellcheck. Please, check the submitted script and try again."
        )
        for i, result in enumerate(shellcheck_results):
            result_str.append(f"Issue {i + 1}")
            result_str.append(f"* Severity level: {result['level']}")
            result_str.append(f"* Message: '{result['message']}'")
            result_str.append(f"* Issue location: line {result['line']}")
        return "\n".join(result_str)

    async def _run_shellcheck(
        self,
        script: str,
    ):
        # Use a here-document to pipe the script directly to shellcheck via stdin
        # Note: shellcheck exits with non-zero when it finds issues, so we use add_to_history=False
        shellcheck_command = f"shellcheck -f json - << 'SHELLCHECK_EOF'\n{script}\nSHELLCHECK_EOF"
        result = await self._execute_bash_command(shellcheck_command, add_to_history=False)
        return result[0]

    async def run_shellcheck(
        self,
        script: str = Field(description="The script to be analyzed by shellcheck."),
    ):
        """Runs shellcheck, a syntax analysis tool for shell scripts that can highlight errors, on a given script."""
        return await self._get_readable_shellcheck_outputs(script)

    def get_tools(self, *args, **kwargs) -> List[BaseTool]:
        return [StructuredTool.from_function(coroutine=self.run_shellcheck)]
