# Prompts for rebench-setup workflow

# Stage 1: Prompt to identify installation-related files
LIST_FILES_PROMPT = """You are tasked with identifying files that likely contain installation
instructions for a GitHub repository.
Repository: {repo_name}
Below is a list of files in the repository that may be helpful for
understanding the installation and setup process:
{files_tree}
Please analyze this list and identify the files that are most likely to
contain information about:
* Installation instructions
* Dependencies or requirements
* Setup procedures
* Development environment configuration
* Testing setup
Think step by step:
* Identify README files, as they often contain installation instructions.
* Look for setup.py, pyproject.toml, requirements.txt, environment.yml.
* Consider files in directories like \\texttt{{docs}} that might contain
installation guides.
* Look for test configuration files that might help understand how to run
tests.
* Consider \\textbf{{only}} files from the list provided above.
* Prioritize files in the repository root (top-level directory).
* Only include files from subdirectories if they are clearly relevant to
installation or setup.
Return **only** a JSON array containing the paths to the most relevant
files for installation and setup. Include only files that are directly
relevant to the tasks above. Sort the files from most to least relevant
and limit your response to no more than 10 files, preferring fewer
files that are truly essential.
For example:
[
"README.md",
"setup.py",
"requirements.txt"
]"""

# Stage 2: Prompt to extract installation recipe and generate bash script
GENERATE_SETUP_PROMPT = """You are tasked with extracting detailed installation instructions from the
following repository files. Repository: {repo_name}
Please analyze the content of these files and extract comprehensive
installation instructions: {rendered}
First, think step by step. After your analysis, return your findings in the
following JSON format:
{{
"python": "3.9",
"packages": "requirements.txt",
"install": "pip install -e .[dev]",
"test_cmd": "pytest --no-header -rA --tb=line --color=no -p no:
cacheprovider -W ignore::DeprecationWarning",
"pre_install": ["apt-get update", "apt-get install -y gcc"],
"reqs_path": ["requirements/base.txt"],
"env_yml_path": ["environment.yml"],
"pip_packages": ["numpy>=1.16.0", "pandas>=1.0.0"]
}}
Here is how this JSON will be used:
```bash
git clone <repo_url> repo
cd repo
git checkout <base_sha>
bash <pre_install>
conda create -n <repo> python=<python> -y
conda activate <repo>
if <packages> == requirements.txt; then
for path in <reqs_path>:
pip install -r $path
elif <packages> == environment.yml; then
for path in <env_yml_path>:
conda env update -f $path
else:
pip install <packages>
pip install <pip_packages>
bash <install>
bash <test_cmd>
```
**IMPORTANT:**
* For the "install" field, always use local install commands like pip
install -e .[dev]
* Do NOT include packages in pip_packages that will be installed by pip
install -e .
* Include only explicitly needed packages in pip_packages.
* reqs_path and env_yml_path must match filenames from the provided files
(e.g., [File: filename]).
* If "packages" is requirements.txt, you must provide at least one
reqs_path.
* Add relevant test frameworks to pip_packages (e.g., pytest, nose).
* Use -y in all conda commands.
* Prefer direct and specific pytest commands over general wrappers.
* Avoid test commands with placeholders like {{test_name}}.
* If a Makefile runs tests, extract the actual test command (e.g., pytest)
.
You must ensure the final JSON includes required fields (, install,
test_cmd), and optionally packages, pre_install, reqs_path,
env_yml_path, pip_packages if relevant.

**CRITICAL: After providing the JSON, you MUST generate a complete bash script that implements the installation recipe described above. The bash script should:**

1. Use the JSON fields to construct the actual installation commands
2. Handle all the conditional logic (requirements.txt vs environment.yml vs direct packages)
3. Include proper error handling and echo statements for each step
4. Use the repository name for the conda environment
5. Execute the pre_install commands if present
6. Install dependencies based on the packages field
7. Run the install command

Return your response in this format. Replace the JSON and bash script placeholders with the actual content:
```json
{{
  "python": "3.9",
  "packages": "requirements.txt",
  "install": "pip install -e .[dev]",
  "pre_install": ["apt-get update", "apt-get install -y gcc"],
  "reqs_path": ["requirements/base.txt"],
  "env_yml_path": ["environment.yml"],
  "pip_packages": ["numpy>=1.16.0", "pandas>=1.0.0"]
}}
```

```bash
#!/bin/bash
set -e

# TODO: replace with your actual data
python="3.9"
packages="requirements.txt"
install="pip install -e .[dev]"
pre_install=("apt-get update" "apt-get install -y gcc")
reqs_path=("requirements/base.txt")
env_yml_path=("environment.yml")
pip_packages=("numpy>=1.16.0" "pandas>=1.0.0")

echo "Starting installation for repository: {repo_name}"

# Pre-install commands
if [ ! -z "$pre_install" ]; then
  echo "Running pre-install commands..."
  for cmd in "${{pre_install[@]}}"; do
    echo "Executing: $cmd"
    eval "$cmd"
  done
fi

# Create conda environment
echo "Creating conda environment: {repo_name}"
conda create -n {repo_name} python=${{python}} -y

# Activate environment
echo "Activating conda environment"
conda activate {repo_name}

# Install dependencies based on packages field
if [ "$packages" = "requirements.txt" ]; then
  echo "Installing from requirements files..."
  for path in "${{reqs_path[@]}}"; do
    echo "Installing from: $path"
    pip install -r "$path"
  done
elif [ "$packages" = "environment.yml" ]; then
  echo "Installing from environment files..."
  for path in "${{env_yml_path[@]}}"; do
    echo "Installing from: $path"
    conda env update -f "$path"
  done
else
  echo "Installing direct package: $packages"
  pip install "$packages"
fi

# Install additional pip packages
if [ ! -z "$pip_packages" ]; then
  echo "Installing additional pip packages..."
  for pkg in "${{pip_packages[@]}}"; do
    echo "Installing: $pkg"
    pip install "$pkg"
  done
fi

# Run install command
echo "Running install command: $install"
eval "$install"
echo "Installation completed successfully."
```

Base your reasoning on all provided files and return both the JSON and bash script in the exact format shown above.""" 