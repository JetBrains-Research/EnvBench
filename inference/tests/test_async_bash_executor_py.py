import asyncio
import os
from textwrap import dedent

import pytest
from dotenv import load_dotenv
from env_setup_utils.repo_downloader import RepoDownloader

from src.async_bash_executor import AsyncBashExecutor

load_dotenv()

docker_image = "ghcr.io/waleko/envsetup-python:latest"
jvm_docker_image = "ghcr.io/waleko/envsetup-jvm:latest"


@pytest.mark.asyncio
async def test_pyenv_commands():
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=120,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )

    from src.toolkits import PythonBashTerminalToolkit

    toolkit = await PythonBashTerminalToolkit.create(bash_executor=bash_executor)
    await toolkit.execute_bash_command(command="pyenv shell 3.11.7", reason="to set Python version to 3.11.7")
    result = await toolkit.execute_bash_command(command="python --version", reason="to check Python version")
    assert "Python 3.11.7" in result
    await toolkit.clean()


@pytest.mark.asyncio
async def test_truncating_output():
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=120,
        max_num_chars_bash_output=2,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )
    result = await bash_executor.execute_bash_command(f"echo {'blahblah' * 10000}")
    assert result[1] == 0
    assert len(result[0]) == 2 + len("\n\n[... 0 lines skipped ...]\n\n")

    result = await bash_executor.execute_bash_command(f"echo {'blahblah' * 10000} && exit 123")
    assert result[1] == 123
    assert len(result[0]) == 2 + len("ERROR: Could not execute given command\n") + len(
        "\n\n[... 0 lines skipped ...]\n\n"
    ) + len("\n")
    await bash_executor.clean()


@pytest.mark.asyncio
async def test_already_downloaded_repo(tmp_path):
    if not os.path.exists(
        f"/{os.path.expanduser('~')}/tmp/repos/Lightning-AI__litgpt@0c609550dfb61699ee513defd64da64634ee3788"
    ):
        repo_downloader = RepoDownloader(
            hf_name="JetBrains-Research/EnvBench",
            output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
            language="python",
        )
        repo_downloader.download(repo_name="Lightning-AI/litgpt", commit_sha="0c609550dfb61699ee513defd64da64634ee3788")

    original_contents = set(
        os.listdir(
            f"/{os.path.expanduser('~')}/tmp/repos/Lightning-AI__litgpt@0c609550dfb61699ee513defd64da64634ee3788"
        )
    )

    # just messing up repo contents somehow idk
    for file in os.listdir(
        f"/{os.path.expanduser('~')}/tmp/repos/Lightning-AI__litgpt@0c609550dfb61699ee513defd64da64634ee3788"
    ):
        if not os.path.isdir(
            f"/{os.path.expanduser('~')}/tmp/repos/Lightning-AI__litgpt@0c609550dfb61699ee513defd64da64634ee3788/{file}"
        ):
            os.remove(
                f"/{os.path.expanduser('~')}/tmp/repos/Lightning-AI__litgpt@0c609550dfb61699ee513defd64da64634ee3788/{file}"
            )
    with open(
        f"/{os.path.expanduser('~')}/tmp/repos/Lightning-AI__litgpt@0c609550dfb61699ee513defd64da64634ee3788/file.txt",
        "w",
    ) as f:
        f.write("123")

    # repodownloader in bashexecutor should clean up before starting work
    bash_executor = await AsyncBashExecutor.create(
        repository="Lightning-AI/litgpt",
        revision="0c609550dfb61699ee513defd64da64634ee3788",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=120,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )
    result = await bash_executor.execute_bash_command("find . -maxdepth 1")
    docker_contents = {content[len("./") :] for content in result[0].split("\n") if content != "."}
    assert docker_contents == original_contents
    await bash_executor.clean()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_apt_get():
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=120,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )
    result = await bash_executor.execute_bash_command("apt-get update")
    assert result[1] == 0
    result = await bash_executor.execute_bash_command("apt-get install -y -qq sl")
    assert result[1] == 0
    await bash_executor.clean()


@pytest.mark.slow
@pytest.mark.skip
@pytest.mark.timeout(300 * 3 + 60)
@pytest.mark.asyncio
async def test_pyenv():
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=300,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )
    # trying to install Python that is not installed in the Docker image by default
    result = await bash_executor.execute_bash_command("pyenv install 3.10.12")
    assert result[1] == 0
    # trying to install Python that is installed in the Docker image by default with -f flag
    result = await bash_executor.execute_bash_command("pyenv install -vf 3.10.13")
    assert result[1] == 0
    # trying to install Python that is installed in the Docker image by default
    # now works because we add -f flag by default
    result = await bash_executor.execute_bash_command("pyenv install 3.10.13")
    assert result[1] == 0
    await bash_executor.clean()


@pytest.mark.asyncio
async def test_timeout_container_restart_multiple_commands():
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=5,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )

    commands = ["sleep 6", "sleep 0.0001", "sleep 6", "sleep 0.0001"]
    results = await asyncio.gather(*(bash_executor.execute_bash_command(command) for command in commands))
    assert results[1][1] == 0 and results[-1][1] == 0
    await bash_executor.clean()


@pytest.mark.asyncio
async def test_timeout_container_restart():
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=5,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )

    await bash_executor.execute_bash_command("sleep 6")
    # command after timeout should work as expected
    result, exit_code = await bash_executor.execute_bash_command("ls")
    expected_files = {
        "environments",
        "examples",
        "LICENSE",
        "planning_library",
        "poetry.lock",
        "pyproject.toml",
        "README.md",
    }
    assert set(_ for _ in result.split()) == expected_files
    assert exit_code == 0
    await bash_executor.clean()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [60, None])
async def test_python_packages(timeout):
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=timeout,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )

    result = await bash_executor.execute_bash_command("conda --version")
    assert "conda" in result[0]
    assert result[1] is None or result[1] == 0

    result = await bash_executor.execute_bash_command("poetry --version")
    assert "Poetry" in result[0]
    assert result[1] is None or result[1] == 0

    result = await bash_executor.execute_bash_command("pipenv --version")
    assert "pipenv" in result[0]
    assert result[1] is None or result[1] == 0

    result = await bash_executor.execute_bash_command("uv --version")
    assert "uv" in result[0]
    assert result[1] is None or result[1] == 0

    result = await bash_executor.execute_bash_command("pip --version")
    assert "pip" in result[0]
    assert result[1] is None or result[1] == 0

    result = await bash_executor.execute_bash_command("pyenv --version")
    assert "pyenv" in result[0]
    assert result[1] is None or result[1] == 0

    await bash_executor.clean()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [60, None])
async def test_repo_structure(timeout):
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=timeout,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )

    result = await bash_executor.execute_bash_command("ls")
    expected_files = {
        "environments",
        "examples",
        "LICENSE",
        "planning_library",
        "poetry.lock",
        "pyproject.toml",
        "README.md",
    }
    assert set(_ for _ in result[0].split()) == expected_files
    assert result[1] is None or result[1] == 0

    result = await bash_executor.execute_bash_command("cat pyproject.toml")
    assert (
        result[0]
        == """[tool.poetry]
name = "planning-library"
version = "0.1.3"
description = "LangChain-based library with planning algorithms for AI Agents."
authors = ["Alexandra Eliseeva <alexandra.eliseeva@jetbrains.com>"]
license = "MIT"
readme = "README.md"
packages = [
    { include = "planning_library" },
    { include = "planning_library/py.typed" },
]
exclude = [
]

[tool.poetry.dependencies]
python = ">=3.9,<4.0"
langchain = "^0.1.4"
langchain-core = "^0.1.30"
langgraph = "^0.0.26"
gymnasium = "^0.29.1"
urllib3 = "<1.27"

[tool.poetry.group.examples.dependencies]
langchain-experimental = "^0.0.49"
langchain-openai = "^0.0.5"
jupyter = "^1.0.0"
pandas = "^2.0.3"
matplotlib = "^3.7.2"
seaborn = "^0.12.2"
gymnasium = {extras = ["toy-text"], version = "^0.29.1"}
moviepy = "^1.0.3"
alfworld = {extras = ["full"], version = "^0.3.3"}

[tool.poetry.group.dev.dependencies]
isort = "^5.12.0"
mypy = "^1.5.0"
pytest = "^7.4.0"
ruff = "^0.3.2"
pyright = "^1.1.368"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
extend-select = ["I"]

[tool.isort]
profile = "black"
force_sort_within_sections = true
order_by_type = true

[tool.mypy]
python_version = "3.9"

[[tool.mypy.overrides]]
module = []
ignore_missing_imports = true"""
    )
    assert (
        bash_executor.commands_history[-1]["exit_code"] is None or bash_executor.commands_history[-1]["exit_code"] == 0
    )

    await bash_executor.clean()


@pytest.mark.timeout(305)
@pytest.mark.asyncio
async def test_timeout():
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=5,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )

    await bash_executor.execute_bash_command("sleep 100000000")
    await bash_executor.clean()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [60, None])
async def test_multiple_commands_consistency(timeout):
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=timeout,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )

    result = await bash_executor.execute_bash_command("export TEST_VAR=hello")
    assert result[1] == 0
    result = await bash_executor.execute_bash_command("echo $TEST_VAR")
    assert result[0] == "hello"
    assert result[1] == 0
    await bash_executor.clean()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [60, None])
async def test_multiple_commands_consistency_revert(timeout):
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=timeout,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )

    result = await bash_executor.execute_bash_command("export TEST_VAR=hello")
    assert result[1] == 0
    # command that exits the execution instance
    result = await bash_executor.execute_bash_command("exit 123")
    assert result[1] == 123
    # state changes from commands before exit are still accessible
    result = await bash_executor.execute_bash_command("echo $TEST_VAR")
    assert result[0] == "hello"
    assert result[1] == 0
    await bash_executor.clean()


@pytest.mark.asyncio
async def test_concurrent_execution():
    """
    Test that commands are properly serialized.
    LangGraph's ToolNode uses asyncio.gather to execute commands concurrently if a message contains multiple tool calls.
    Therefore, we need to ensure that commands don't interfere with each other.
    """

    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=60,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="python",
    )

    # Test concurrent execution to verify commands are properly serialized
    base_cmd = dedent("""
        if [ -f /tmp/test.lock ]; then
            echo "Lock file exists, commands are racing!"
            exit 1
        fi
        touch /tmp/test.lock
        sleep 2
        rm /tmp/test.lock
        echo "cmd{} done"
    """)

    # Execute commands concurrently (gather like ToolNode)
    results = await asyncio.gather(*(bash_executor.execute_bash_command(base_cmd.format(i)) for i in range(1, 4)))

    # Check all commands completed successfully
    for i, (output, exit_code) in enumerate(results, 1):
        assert exit_code == 0
        assert "commands are racing" not in output
        assert f"cmd{i} done" in output

    await bash_executor.clean()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [60, None])
async def test_jvm_packages(timeout):
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=jvm_docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=timeout,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="jvm",
    )

    # Test Java
    result = await bash_executor.execute_bash_command("java -version")
    assert result[1] is None or result[1] == 0

    # Test Maven
    result = await bash_executor.execute_bash_command("mvn --version")
    assert "Apache Maven" in result[0]
    assert result[1] is None or result[1] == 0

    # Test Gradle
    result = await bash_executor.execute_bash_command("gradle --version")
    assert "Gradle" in result[0]
    assert result[1] is None or result[1] == 0

    # Test XML tools
    result = await bash_executor.execute_bash_command("xmlstarlet --version")
    assert result[1] is None or result[1] == 0

    # Test SDKMAN
    result = await bash_executor.execute_bash_command("source ${SDKMAN_DIR}/bin/sdkman-init.sh && sdk version")
    assert "SDKMAN" in result[0], result[0]
    assert result[1] is None or result[1] == 0

    await bash_executor.clean()


@pytest.mark.asyncio
async def test_jvm_bash_terminal_toolkit(timeout=60):
    bash_executor = await AsyncBashExecutor.create(
        repository="JetBrains-Research/planning-library",
        revision="a4282f30dc5db17c6a68715295f3f7d77b766b0d",
        image=jvm_docker_image,
        error_message=None,
        env_vars={},
        repository_workdir=True,
        container_start_timeout=300,
        bash_timeout=timeout,
        max_num_chars_bash_output=16000,
        hf_name="JetBrains-Research/EnvBench",
        output_dir=f"/{os.path.expanduser('~')}/tmp/repos",
        language="jvm",
    )

    from src.toolkits.bash_terminal_jvm import JVMBashTerminalToolkit

    toolkit = await JVMBashTerminalToolkit.create(bash_executor=bash_executor)

    # Test Maven through toolkit
    result = await toolkit.execute_bash_command(
        command="mvn --version", reason="to verify Maven is installed and accessible"
    )
    assert "Apache Maven" in result

    # Test Gradle through toolkit
    result = await toolkit.execute_bash_command(
        command="gradle --version", reason="to verify Gradle is installed and accessible"
    )
    assert "Gradle" in result

    # Test SDKMAN through toolkit
    result = await toolkit.execute_bash_command(
        command="sdk version", reason="to verify SDKMAN is installed and accessible"
    )
    assert "SDKMAN" in result

    # Test environment variables
    result = await toolkit.execute_bash_command(command="echo $JAVA_HOME", reason="to verify JAVA_HOME is properly set")
    assert "/root/.sdkman/candidates/java" in result

    # Test toolkit command history
    assert len(toolkit.commands_history) == 5
    for cmd in toolkit.commands_history:
        assert isinstance(cmd, dict)
        assert "command" in cmd
        assert "exit_code" in cmd

    await toolkit.clean()
