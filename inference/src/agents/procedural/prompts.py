"""Prompts for the procedural environment setup agent."""

from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

# Load Dockerfiles
python_dockerfile_path = Path(__file__).parents[4] / "env_setup_utils" / "scripts" / "python.Dockerfile"
jvm_dockerfile_path = Path(__file__).parents[4] / "env_setup_utils" / "scripts" / "jvm.Dockerfile"
python_dockerfile = python_dockerfile_path.read_text()
jvm_dockerfile = jvm_dockerfile_path.read_text()
# Load baseline scripts
python_baseline_path = Path(__file__).parents[4] / "evaluation" / "scripts" / "python_baseline.sh"
jvm_baseline_path = Path(__file__).parents[4] / "evaluation" / "scripts" / "jvm_baseline.sh"
python_baseline = python_baseline_path.read_text()
jvm_baseline = jvm_baseline_path.read_text()

PYTHON_SETUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Your task is to generate a bash script that will set up a Python development environment for a repository mounted in the current directory.
You will be provided with repository context. Follow the build instructions to generate the script.

A very universal script might look like this:
```bash
{baseline_script}
```
However, your job is to make a script more tailored to the repository context.
It will be only run on a single repository mounted in the current directory that you have information about.
The script must not be universal but setup the environment just for this repository.
Avoid using universal if-else statements and try to make the script as specific as possible.

The script should:
1. Install the correct Python version based on repository requirements
2. Install all project dependencies from requirements.txt/setup.py/pyproject.toml
3. Install any required system packages

For reference, the script will run in this Docker environment, so most of the tools you need will be available:
```
{dockerfile}
```

IMPORTANT:
- Generate ONLY a bash script - you cannot interact with the system
- The script must be non-interactive (use -y flags where needed)
- Base all decisions on the provided repository context. Follow the context instructions.
- Don't use sudo - the script will run as root
- if you use pyenv install, please use -f flag to force the installation. For example: `pyenv install -f $PYTHON_VERSION`
- The script must be enclosed in ```bash``` code blocks"""),
    ("user", """Build Instructions:
{build_instructions}

Repository Context:
{context}

Generate a complete bash script that will set up this Python environment.
The script must be enclosed in ```bash``` code blocks, it can rely on the tools available in the Docker environment.""")
])

JVM_SETUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Your task is to generate a bash script that will set up a JVM development environment.
You will be provided with repository context and build instructions. Follow the build instructions to generate the script.

A very universal script might look like this:
```bash
{baseline_script}
```
However, your job is to make a script more tailored to the repository context.
It will be only run on a single repository mounted in the current directory that you have information about.
The script must not be universal but setup the environment just for this repository.
Avoid using universal if-else statements and try to make the script as specific as possible.

The script should:
1. Install the correct Java version based on repository requirements
2. Install and configure build tools (Maven/Gradle)
3. Install all project dependencies
4. Install any required system packages using apt-get

The environment has the following tools available:
- sdk for Java version management
- Maven and Gradle for builds

For reference, the script will run in this Docker environment, so most of the tools are already installed:
```
{dockerfile}
```

IMPORTANT:
- Generate ONLY a bash script - you cannot interact with the system
- The script must be non-interactive (use -y flags where needed)
- Base all decisions on the provided repository context. Follow the instructions in the context.
- Don't use sudo. The script will run as root
- The script must be enclosed in ```bash``` code blocks"""),
    ("user", """Build Instructions:
{build_instructions}

Repository Context:
{context}

Generate a complete bash script that will set up this JVM environment.
The script must be enclosed in ```bash``` code blocks, it can rely on the tools available in the Docker environment.""")
])

def get_python_setup_prompt(state: dict) -> str:
    """Get the prompt for Python environment setup."""

    print(state)
    return PYTHON_SETUP_PROMPT.format(
        build_instructions=state["build_instructions"],
        context=state["context"],
        dockerfile=python_dockerfile,
        baseline_script=python_baseline
    )

def get_jvm_setup_prompt(state: dict) -> str:
    """Get the prompt for JVM environment setup."""
    return JVM_SETUP_PROMPT.format(
        build_instructions=state["build_instructions"],
        context=state["context"],
        dockerfile=jvm_dockerfile,
        baseline_script=jvm_baseline
    )    
