# Prompts for rebench-readonly workflow
import requests

python_dockerfile = requests.get(
    "https://raw.githubusercontent.com/JetBrains-Research/EnvBench/main/dockerfiles/python.Dockerfile"
).text

# ReAct agent prompt for exploring repository and understanding setup
REACT_AGENT_PROMPT = """# SETTING

You are an intelligent AI agent with the goal of understanding and analyzing a Python repository to determine how to set up its development environment.
The repository is already downloaded and located at /data/project in your working directory, and you can access all files and folders through bash commands.
You will interact with the environment exclusively by sending commands to the terminal.

**IMPORTANT**: You are working in a READ-ONLY filesystem environment. You can explore, investigate, and analyze the repository, but you CANNOT modify any files or install packages. Your goal is to understand the setup requirements and generate a bash script that would configure the environment when run elsewhere.

Repository: {repo_name}

# BASE ENVIRONMENT

The environment has standard Unix commands available (e.g. ls, grep, find, cat, etc.) for repository exploration.
You can read files, examine directory structures, and analyze configuration files.
You CANNOT install packages, modify files, or run interactive commands.

# ENVIRONMENT SETUP ANALYSIS GUIDELINES

1. **Repository Exploration**: Start by examining the repository structure to understand:
   - Root folder contents and any subfolders (src, lib, etc.)
   - Dependency definition files (requirements.txt, pyproject.toml, setup.py, setup.cfg, Pipfile, etc.)
   - Documentation files (README, INSTALL, docs/, etc.)
   - Configuration files (.python-version, runtime.txt, etc.)

2. **Python Version Requirements**: 
   - Check for Python version requirements in documentation or config files
   - Look for .python-version, runtime.txt, or version specifications in setup files
   - Note any specific version constraints

3. **Dependency Manager Detection**:
   - Identify whether the project uses pip, Poetry, pipenv, or other tools
   - Look for: requirements.txt, pyproject.toml, Pipfile, setup.py, setup.cfg
   - Analyze the dependency structure and installation approach

4. **Installation Process Analysis**:
   - Follow any specific installation instructions found in README or documentation
   - Identify system-level dependencies that might need to be installed
   - Understand how Python dependencies should be installed
   - Note any dev dependencies and optional extras
   - Check if the project should be installed locally with `pip install -e .`

5. **Environment Configuration**:
   - Understand any environment activation requirements
   - Note any specific configuration files or environment variables
   - Identify build or compilation steps if applicable

# WORKFLOW

1. **Explore**: Use terminal commands to understand the repository structure and requirements
2. **Analyze**: Examine configuration files and documentation to understand setup needs
3. **Document**: Note all requirements, dependencies, and setup steps
4. **Plan**: Determine the complete setup process that would be needed

# YOUR RESPONSE

Always respond with your reasoning followed by the terminal command to execute.
The command will be executed and you'll receive the output.
Continue working iteratively until you have a complete understanding of the setup requirements.

# ENVIRONMENT RESPONSE

The environment will provide you with command output or error messages, followed by a shell prompt.

# USEFUL COMMANDS

* Start with `ls -la` to see the repository structure
* Use `find . -name "*.txt" -o -name "*.toml" -o -name "*.py" -o -name "*.md"` to find key files
* Read README files first - they often contain crucial setup instructions
* Examine requirements.txt, pyproject.toml, setup.py for dependency information
* Use `cat` to read file contents and understand configuration

# COMMON PITFALLS TO AVOID

* Don't try to install packages or modify files - this is a read-only environment
* Don't use interactive commands that wait for user input
* Focus on understanding the setup requirements, not executing them
* Pay attention to system dependencies mentioned in documentation
* Don't assume default configurations - check all relevant files

Remember: You are in a READ-ONLY environment. Your goal is to understand and document the setup process, not execute it."""

# Optional summarization prompt
SUMMARIZATION_PROMPT = """Based on your exploration of the repository {repo_name}, provide a comprehensive summary of the setup requirements and environment configuration needed.

Your summary should include:

1. **Project Structure**: Key directories and files that define the project
2. **Python Version Requirements**: Any specific Python version constraints
3. **Dependency Management**: Which tool is used (pip, Poetry, pipenv, etc.) and key dependencies
4. **System Dependencies**: Any system packages that need to be installed via apt-get
5. **Installation Process**: Step-by-step setup instructions found in documentation
6. **Environment Configuration**: Any special configuration, environment variables, or activation requirements
7. **Build/Compilation Steps**: Any additional setup steps beyond dependency installation

This summary will be used to generate a complete bash script for setting up the development environment.

Focus on being comprehensive and specific - include exact version numbers, package names, and commands where possible."""

# Common bash script generation content
BASH_SCRIPT_COMMON = """

**CRITICAL DOCKER ENVIRONMENT CONSTRAINTS**:
The script will run in a Docker container with the following base configuration:
```
{dockerfile}
```

**IMPORTANT**: The Docker image already has Python, pyenv, pip, git, and common development tools pre-installed and configured. DO NOT attempt to reinstall these tools as it will likely cause errors. Use the existing installations.

**REPOSITORY LOCATION**: The repository is already downloaded and located at /data/project. Do not include any git clone commands in the script.

The script should:
1. Install ONLY the system dependencies that are NOT already in the base image using apt-get (with -y flags)
2. Use the existing pyenv installation to set up the correct Python version if needed
3. Install all project dependencies using the appropriate package manager
4. Handle any special environment configuration or activation
5. Include proper error handling with `set -e` - the script MUST run without any errors
6. Use echo statements to show progress
7. Be non-interactive (use -y flags where needed)
8. Install the project in development mode if applicable (`pip install -e .`)

**ERROR-FREE EXECUTION REQUIREMENTS**:
- The script MUST run with `set -e` and exit on any error
- Test all commands mentally before including them
- Avoid redundant installations of already-available tools
- Use `which` or `command -v` to check if tools exist before using them
- Handle edge cases gracefully

**IMPORTANT**: 
- First, explain your reasoning for the setup approach based on the repository analysis
- Then generate ONLY a single markdown code block with the bash script
- The script must be enclosed in ```bash``` tags
- The script should be complete and self-contained
- Ensure every command will succeed in the Docker environment

Generate the script now."""

# Final bash script generation prompt (when summarization is enabled)
BASH_SCRIPT_PROMPT = (
    "Based on the exploration and analysis of repository {repo_name}, generate a complete bash script that sets up the development environment from scratch.\n\nRepository Context: {summary}"
    + BASH_SCRIPT_COMMON
)

# Final bash script generation prompt (when summarization is disabled)
BASH_SCRIPT_NO_SUMMARY_PROMPT = (
    "Based on the exploration of repository {repo_name}, please generate a complete bash script that sets up the development environment from scratch."
    + BASH_SCRIPT_COMMON
)
