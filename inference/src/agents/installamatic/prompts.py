from pathlib import Path
from typing import List

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agents.installamatic.state_schema import (
    InstallamaticBuildConfigurable,
    InstallamaticBuildState,
    InstallamaticSearchConfigurable,
    InstallamaticSearchState,
)

# Load Dockerfiles
python_dockerfile_path = Path(__file__).parents[4] / "env_setup_utils" / "scripts" / "python.Dockerfile"
jvm_dockerfile_path = Path(__file__).parents[4] / "env_setup_utils" / "scripts" / "jvm.Dockerfile"
python_dockerfile = python_dockerfile_path.read_text()
jvm_dockerfile = jvm_dockerfile_path.read_text()

SEARCH_SYSTEM_PROMPT_PYTHON = """I want to write a shell script to build the <REPO_NAME> project.
Your task is to search the given repository using the provided tools and collect all files that contain documentation related to environment setup, installing dependencies and running tests.
You musnt try to install the repository using pip. `pip install <REPO_NAME>` is NOT what we want to find, this is INCORRECT and IRRELEVANT to building the project from source, which is our goal.
Such content could be found in files with names like "testing", "contributing", "setup", etc.
To reiterate, a file that makes explicit mention of **how to install the project's dependencies** is considered relevant. Anything else should be ignored.
Your focus should be on natural langauge documents. DO NOT attempt to read or register python files, for example.
However, pay attention to configuration files that might contain relevant information:
- requirements.txt, pyproject.toml, setup.py, setup.cfg are used with pip
- poetry.lock and pyproject.toml are used with Poetry
- Makefile
Note that the presence of pyproject.toml is not enough to determine the dependency manager! Check the build system specified in pyproject.toml (setuptools means dependencies could be installed with pip)
and the presence of poetry.lock.

Whenever called, first plan your next move concisely in only one or two sentences, then use one of the provided tools to retrieve additional information about the repo's contents.
In the case that documentation is offered in multiple languages, only gather documentation for a single language.

Whenever you find a documentation file, you will submit it using the `submit_documentation` tool, then continue your search. The whole file is submitted at once, so even if a document contains multiple relevant sections, you only need to submit it once.
Once you are confident that you have found all documentation files in the repository, use the `finished_search` tool to move on to your next task.

Here are the contents of the repository's root directory (`.`):
<CONTENTS>"""

SEARCH_SYSTEM_PROMPT_JVM = """I want to write a shell script to build the <REPO_NAME> project. It is either a Java or a Kotlin project and it uses either Gradle or Maven.
Your task is to search the given repository using the provided tools and collect all files that contain documentation related to environment setup, installing dependencies and running tests.
Such content could be found in files with names like "testing", "contributing", "setup", etc.
To reiterate, a file that makes explicit mention of **how to install the project's dependencies** is considered relevant. Anything else should be ignored.
Your focus should be on natural langauge documents. DO NOT attempt to read or register code files, for example.
Pay attention to Gradle or Maven configuration files that might contain relevant information.

Whenever called, first plan your next move concisely in only one or two sentences, then use one of the provided tools to retrieve additional information about the repo's contents.
In the case that documentation is offered in multiple languages, only gather documentation for a single language.

Whenever you find a documentation file, you will submit it using the `submit_documentation` tool, then continue your search. The whole file is submitted at once, so even if a document contains multiple relevant sections, you only need to submit it once.
Once you are confident that you have found all documentation files in the repository, use the `finished_search` tool to move on to your next task.

Here are the contents of the repository's root directory (`.`):
<CONTENTS>"""


async def get_installamatic_search_prompt(state: InstallamaticSearchState, config: RunnableConfig) -> List[BaseMessage]:
    configurable: InstallamaticSearchConfigurable = config.get("configurable", {}).get("search", {})
    language = configurable["language"]
    get_directory_contents_tool = configurable["get_directory_contents_tool"]

    root_contents = await get_directory_contents_tool.ainvoke(".", config=config)

    if language == "python":
        system_prompt = SEARCH_SYSTEM_PROMPT_PYTHON.replace("<REPO_NAME>", state.get("repository", "")).replace(
            "<CONTENTS>", root_contents
        )
    elif language == "jvm":
        system_prompt = SEARCH_SYSTEM_PROMPT_JVM.replace("<REPO_NAME>", state.get("repository", "")).replace(
            "<CONTENTS>", root_contents
        )
    else:
        raise ValueError(f"Unknown language {language}.")
    return [SystemMessage(content=system_prompt)]


BUILD_SYSTEM_PROMPT_PYTHON = """I want to write a shell script to place inside this repo that will set up a development environment and install any dependencies.
Remember, instructions such as `pip install <REPO_NAME>` are NOT helpful. I want to build the project from source.
Using the gathered files, collect and summarise any information that may help me. The gathered files, which you now have access to are:
<FILES>

You may use the provided tools to reinspect the contents of these files if you wish, but you can not access any files other than these.

Information that might be useful to include in the summary:
- Are there any restrictions on Python version? Which version is required?
- Are there any additional system packages that should be installed before installing dependencies?
- What dependency manager is used in the given repository (pip or Poetry)? What is the path to the corresponding configuration files?
- Are there any additional considerations (e.g., mention optional dependency groups in the configuration files if they are present)?

Please, do not try to include the shell script itself in your summary, I will write it in a separate step.
 
Once you are done, use the `submit_summary` to give a summary of the information you found.
"""

BUILD_SYSTEM_PROMPT_JVM = """I want to write a shell script to place inside this repo that will set up a development environment and install any dependencies.
Using the gathered files, collect and summarise any information that may help me. The gathered files, which you now have access to are:
<FILES>

You may use the provided tools to reinspect the contents of these files if you wish, but you can not access any files other than these.

- Are there any restrictions on Java version? Which version is required?
- Are there any additional system packages that should be installed before installing dependencies?
- What build tool is used in the given repository (Gradle or Maven)? What is the path to the corresponding configuration files?
- Are there any additional considerations?

Please, do not try to include the shell script itself in your summary, I will write it in a separate step.

Once you are done, use the `submit_summary` to give a summary of the information you found.
"""


async def get_installamatic_build_prompt(state: InstallamaticBuildState, config: RunnableConfig) -> List[BaseMessage]:
    configurable: InstallamaticBuildConfigurable = config.get("configurable", {}).get("build", {})
    language = configurable["language"]

    if language == "python":
        system_prompt = BUILD_SYSTEM_PROMPT_PYTHON.replace("<REPO_NAME>", state.get("repository", "")).replace(
            "<FILES>", "\n".join(f"- {file}" for file in state.get("documentation", []))
        )
    elif language == "jvm":
        system_prompt = BUILD_SYSTEM_PROMPT_JVM.replace("<REPO_NAME>", state.get("repository", "")).replace(
            "<FILES>", "\n".join(f"- {file}" for file in state.get("documentation", []))
        )
    else:
        raise ValueError(f"Unknown language {language}.")

    return [SystemMessage(content=system_prompt)]


GENERATE_SHELL_SCRIPT_SYSTEM_PROMPT_PYTHON = """Now that you have gathered sufficient information about the repository, your new task is to generate a bash script that will set up a Python development environment for a repository mounted in the current directory.
You have access to repository context from previous interactions.

The script should:
1. Install the correct Python version based on repository requirements
2. Install all project dependencies from requirements.txt/setup.py/pyproject.toml
3. Install any required system packages

For reference, the script will run in this Docker environment, so most of the tools you need will be available:
```
{dockerfile}
```
You DO NOT need to install pip or poetry. Use pyenv to install and configure the shell to use a specific Python version.

IMPORTANT:
- Generate ONLY a bash script - you cannot interact with the system
- The script must be non-interactive (use -y flags where needed)
- Base all decisions on the provided repository context. Follow the context instructions.
- Don't use sudo - the script will run as root
- If you install the dependencies via `pip install .`, please, consider using editable mode: `pip install -e .`
- If you use pyenv install, please use -f flag to force the installation. For example: `pyenv install -f $PYTHON_VERSION`

In your script, make sure that the Python project can be run simply by using the “python” command (which should point to the correct executable). If, for example, you are using Poetry,
you must run “source $(poetry env info --path)/bin/activate” so that the “python” command is associated with the Poetry environment.
Don't run "poetry shell" as it does not work in the Docker container.

Use the `submit_shell_script` function to provide the shell script.
""".format(dockerfile=python_dockerfile)

GENERATE_SHELL_SCRIPT_SYSTEM_PROMPT_JVM = """Now that you have gathered sufficient information about the repository, your new task is to generate a bash script that will set up a JVM development environment for a repository mounted in the current directory.
You have access to repository context from previous interactions.

The script should:
1. Install the correct Java version based on repository requirements
2. Install and configure build tools (Maven/Gradle)
3. Install all project dependencies
4. Install any required system packages using apt-get

The environment has the following tools available:
- sdk for Java version management
- Maven and Gradle for builds

For reference, the script will run in this Docker environment, so most of the tools you need will be available:
```
{dockerfile}
```

IMPORTANT:
- Generate ONLY a bash script - you cannot interact with the system
- The script must be non-interactive (use -y flags where needed)
- Base all decisions on the provided repository context. Follow the instructions in the context.
- Don't use sudo. The script will run as root

Use the `submit_shell_script` function to provide the shell script.
""".format(dockerfile=jvm_dockerfile)


async def get_installamatic_generate_shell_script_prompt(
    state: InstallamaticBuildState, config: RunnableConfig
) -> List[BaseMessage]:
    configurable: InstallamaticBuildConfigurable = config.get("configurable", {}).get("build", {})
    language = configurable["language"]

    if language == "python":
        system_prompt = GENERATE_SHELL_SCRIPT_SYSTEM_PROMPT_PYTHON.replace("<REPO_NAME>", state.get("repository", ""))
    elif language == "jvm":
        system_prompt = GENERATE_SHELL_SCRIPT_SYSTEM_PROMPT_JVM.replace("<REPO_NAME>", state.get("repository", ""))
    else:
        raise ValueError(f"Unknown language {language}.")

    return [SystemMessage(content=system_prompt)]
