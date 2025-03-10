import asyncio
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple, TypedDict

from aiodocker import Docker
from aiodocker.containers import DockerContainer
from aiodocker.exceptions import DockerError
from aiodocker.execs import Exec
from aiodocker.stream import Stream
from env_setup_utils.repo_downloader import RepoDownloader


class CommandExecutionResult(TypedDict):
    command: str
    exit_code: Optional[int]


class AsyncBashExecutor:
    DEFAULT_ERROR: str = "ERROR: Could not execute given command."
    DEFAULT_COMMAND: str = "while true; do sleep 1000; done"

    def __init__(
        self,
        repository: str,
        revision: str,
        image: str,
        command: Optional[str],
        error_message: Optional[str],
        env_vars: Dict[str, str],
        repository_workdir: bool,
        container_start_timeout: int,
        bash_timeout: Optional[int],
        bash_timeout_exit_code: int,
        max_num_chars_bash_output: Optional[int],
        docker_client: Docker,
        container: DockerContainer,
        output_dir: str,
        hf_name: str,
        language: str,
        clear_repo: bool,
        exec_instance: Exec,
        exec_stream: Stream,
    ):
        self.repository = repository
        self.revision = revision

        self.image = image
        self.env_vars = env_vars
        self.repository_workdir = repository_workdir
        self.command = command
        self.error_message = error_message or self.DEFAULT_ERROR

        self.container_start_timeout = container_start_timeout
        self.bash_timeout = bash_timeout
        self.bash_timeout_exit_code = bash_timeout_exit_code
        self.max_num_chars_bash_output = max_num_chars_bash_output
        self.clear_repo = clear_repo

        self.commands_history: List[CommandExecutionResult] = []
        """List of tuples with bash commands and their exit codes."""

        self.client: Docker = docker_client
        self.container: DockerContainer = container

        self.exec_instance: Exec = exec_instance
        self.exec_stream: Stream = exec_stream

        self.output_dir: str = output_dir
        self.hf_name = hf_name
        self.language = language

        self._command_lock = asyncio.Lock()

    @staticmethod
    async def _init_exec_stream(
        container: DockerContainer, repository: str, revision: str, repository_workdir: bool
    ) -> Tuple[Exec, Stream]:
        logging.info(f"[{repository}@{revision}] Starting new exec.")
        exec_instance = await container.exec(
            ["/bin/bash"],
            workdir=os.path.join("/data/project", f"{repository.replace('/', '__')}@{revision}")
            if repository_workdir
            else None,
            stdin=True,
            stdout=True,
            stderr=True,
        )
        exec_stream = exec_instance.start(detach=False)
        await exec_stream._init()
        return exec_instance, exec_stream

    @staticmethod
    def _download_repo(repository: str, revision: str, hf_name: str, output_dir: str, language: str) -> str:
        repo_downloader = RepoDownloader(hf_name=hf_name, output_dir=output_dir, language=language)
        is_downloaded = repo_downloader.download(repo_name=repository, commit_sha=revision)
        if not is_downloaded:
            raise ValueError(f"Unable to download repository {repository}@{revision}.")
        return repo_downloader.get_repo_dir_path(repo_name=repository, commit_sha=revision)

    @classmethod
    async def create(
        cls,
        repository: str,
        revision: str,
        image: str,
        hf_name: str,
        output_dir: str,
        language: str,
        command: Optional[str] = None,
        clear_repo: bool = True,
        bash_timeout_exit_code: int = -123,
        error_message: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        repository_workdir: bool = True,
        container_start_timeout: int = 30,
        bash_timeout: Optional[int] = None,
        max_num_chars_bash_output: Optional[int] = None,
    ) -> "AsyncBashExecutor":
        env_vars = env_vars or {}
        client = Docker()
        try:
            await cls._pull_image(client=client, image=image)
            container = await cls._start_container(
                client=client,
                image=image,
                repository=repository,
                revision=revision,
                command=command,
                env_vars=env_vars,
                timeout=container_start_timeout,
                hf_name=hf_name,
                output_dir=output_dir,
                language=language,
            )

            exec_instance, exec_stream = await cls._init_exec_stream(
                container=container,
                repository=repository,
                revision=revision,
                repository_workdir=repository_workdir,
            )

            return cls(
                repository=repository,
                revision=revision,
                image=image,
                bash_timeout_exit_code=bash_timeout_exit_code,
                error_message=error_message,
                env_vars=env_vars,
                repository_workdir=repository_workdir,
                container_start_timeout=container_start_timeout,
                bash_timeout=bash_timeout,
                max_num_chars_bash_output=max_num_chars_bash_output,
                docker_client=client,
                container=container,
                exec_stream=exec_stream,
                exec_instance=exec_instance,
                output_dir=output_dir,
                hf_name=hf_name,
                language=language,
                clear_repo=clear_repo,
                command=command,
            )
        except Exception:
            await client.close()
            raise

    @staticmethod
    async def _pull_image(client: Docker, image: str) -> None:
        try:
            await client.images.inspect(image)
            logging.info(f"Image '{image}' already exists locally.")
        except DockerError as e:
            if e.status == 404:
                logging.info(f"Pulling image '{image}'...")
                await client.images.pull(image)
                logging.info(f"Image '{image}' pulled successfully.")
            else:
                logging.error(f"Error retrieving image '{image}': {e}")
                raise

    @staticmethod
    async def _start_container(
        client: Docker,
        image: str,
        repository: str,
        revision: str,
        command: Optional[str],
        env_vars: Dict[str, str],
        hf_name: str,
        language: str,
        output_dir: str,
        timeout: int,
    ) -> DockerContainer:
        logging.info(f"[{repository}@{revision}] Downloading repository.")
        local_repo_path = AsyncBashExecutor._download_repo(
            repository=repository,
            revision=revision,
            hf_name=hf_name,
            language=language,
            output_dir=output_dir,
        )
        repository_dir = os.path.basename(local_repo_path)

        default_command = AsyncBashExecutor.DEFAULT_COMMAND.format(
            repository=repository,
            repository_dir=repository_dir,
            revision=revision,
        )
        final_command = (
            ["-c", default_command]
            if command is None
            else ["-c", command.format(repository=repository, repository_dir=repository_dir, revision=revision)]
        )

        container = await client.containers.create(
            {
                "Image": image,
                "Cmd": final_command,
                "Env": [f"{key}={value}" for key, value in env_vars.items()],
                "Entrypoint": "/bin/bash",
                "Detach": True,
                "HostConfig": {"Binds": [f"{local_repo_path}:/data/project/{os.path.basename(local_repo_path)}:rw"]},
            }
        )
        await container.start()
        logging.info(f"[{repository}@{revision}] Starting container {container.id}.")

        start_time = time.time()
        while time.time() - start_time < timeout:
            container_info = await container.show()

            if container_info["State"].get("Status") == "running":
                logging.info(f"[{repository}@{revision}] Container {container.id} started successfully.")
                return container

            elif container_info["State"].get("Status"):
                logs = await container.log(stdout=True, stderr=True)
                logging.error(f"[{repository}@{revision}] Container {container.id} exited on start.")
                logging.error(f"[{repository}@{revision}]  Container logs: {logs}")
                raise RuntimeError("Could not start container.")
            await asyncio.sleep(0.1)

        logging.error(f"[{repository}@{revision}] Container {container.id} failed to start within the timeout period.")
        raise TimeoutError("Could not start container within the timeout period.")

    async def _stop_container(self) -> None:
        try:
            container_info = await self.container.show()
            status = container_info["State"]["Status"]
            if status == "running":
                await self.container.stop()
                logging.info(f"[{self.repository}@{self.revision}] Container {self.container.id} stopped.")
            await self.container.delete(force=True)
            logging.info(f"[{self.repository}@{self.revision}] Container {self.container.id} removed.")
        except DockerError as e:
            logging.error(f"[{self.repository}@{self.revision}] Error stopping container {self.container.id}: {e}")

    async def restart_container(self) -> None:
        try:
            if self.container:
                container_info = await self.container.show()
                if container_info.get("State", {}).get("Running", False):
                    await self._stop_container()
        except DockerError:
            ...

        container = await self._start_container(
            client=self.client,
            image=self.image,
            repository=self.repository,
            revision=self.revision,
            env_vars=self.env_vars,
            hf_name=self.hf_name,
            language=self.language,
            output_dir=self.output_dir,
            timeout=self.container_start_timeout,
            command=self.command,
        )
        self.container = container

        exec_instance, exec_stream = await self._init_exec_stream(
            container=self.container,
            repository=self.repository,
            revision=self.revision,
            repository_workdir=self.repository_workdir,
        )
        self.exec_instance = exec_instance
        self.exec_stream = exec_stream

        for command in self.commands_history:
            if command["exit_code"] == 0:
                output, exit_code = await self._execute_bash_command(command["command"])
                assert exit_code == 0

    async def _execute_bash_command(self, command: str) -> Tuple[str, int]:
        command_id = uuid.uuid4().hex
        end_marker = f"__END_OF_COMMAND_{command_id}__"
        end_marker_bytes = end_marker.encode("utf-8")

        try:
            full_command = f"{command}\nexit_code=$?\necho __EXIT_CODE__ $exit_code\necho {end_marker}\n"
            full_command_bytes = full_command.encode("utf-8")
            await self.exec_stream.write_in(full_command_bytes)

            output = b""
            error = b""
            exit_code = None

            while True:
                try:
                    msg = await asyncio.wait_for(self.exec_stream.read_out(), timeout=self.bash_timeout)
                except asyncio.TimeoutError:
                    msg = None
                    error += b"Timed out."
                    exit_code = self.bash_timeout_exit_code
                    await self.restart_container()
                if msg is None:
                    break
                if msg.stream == 1:
                    output += msg.data
                    if end_marker_bytes in output:
                        break
                elif msg.stream == 2:
                    error += msg.data

            output_decoded = output.decode("utf-8").strip().split(end_marker)[0]

            output_lines = output_decoded.split("__EXIT_CODE__")
            if not exit_code:
                if len(output_lines) > 1:
                    output_decoded, exit_code_line = output_decoded.split("__EXIT_CODE__")
                    output_decoded, exit_code_line = output_decoded.strip(), exit_code_line.strip()
                    exit_code = int(exit_code_line) if exit_code_line.isdigit() else 0
                else:
                    inspect = await self.exec_instance.inspect()
                    inspect_exit_code = inspect.get("ExitCode")
                    if isinstance(inspect_exit_code, int):
                        exit_code = inspect_exit_code
                    else:
                        exit_code = 0
            error_decoded = error.decode("utf-8").strip()

            text = f"stdout:\n{output_decoded}\n\nstderr:\n{error_decoded}"

            return text, exit_code
        except DockerError:
            logging.error(f"[{self.repository}@{self.revision}] Error executing command '{command}'.")
            return self.error_message, 1

    async def execute_bash_command(self, command: str, add_to_history: bool = True) -> Tuple[str, int]:
        """
        Executes a given bash command inside the Docker container asynchronously.
        """
        async with self._command_lock:
            need_to_restart = False
            if not self.container or not self.exec_instance or not self.exec_stream:
                need_to_restart = True
            else:
                try:
                    container_info = await self.container.show()
                    if not container_info.get("State", {}).get("Running", False):
                        need_to_restart = True

                    inspect = await self.exec_instance.inspect()
                    if not inspect.get("Running", False):
                        need_to_restart = True
                except DockerError:
                    need_to_restart = True

            if need_to_restart:
                logging.error(
                    f"[{self.repository}@{self.revision}] Container or exec is not running. Restarting container."
                )
                await self.restart_container()

            output, exit_code = await self._execute_bash_command(command)

        if add_to_history:
            self.commands_history.append({"command": command, "exit_code": exit_code})

        if self.max_num_chars_bash_output is not None and len(output) > self.max_num_chars_bash_output:
            # Leave first half and last half of the output
            first_half = output[: self.max_num_chars_bash_output // 2]
            last_half = output[-self.max_num_chars_bash_output // 2 :]
            lines_skipped = output.count(
                "\n", self.max_num_chars_bash_output // 2, -self.max_num_chars_bash_output // 2
            )
            output = f"{first_half}\n\n[... {lines_skipped} lines skipped ...]\n\n{last_half}"

        if exit_code != 0:
            return f"{self.error_message}\n{output}", exit_code

        return output, exit_code

    async def clean(self):
        try:
            await self._stop_container()
            if self.clear_repo:
                repo_downloader = RepoDownloader(
                    hf_name=self.hf_name, output_dir=self.output_dir, language=self.language
                )
                repo_downloader.clear_repo(repo_name=self.repository, commit_sha=self.revision)
                logging.info(f"[{self.repository}@{self.revision}] Repository removed.")
        except DockerError as e:
            logging.error(f"[{self.repository}@{self.revision}] Error cleaning: {e}")
        finally:
            await self.client.close()
            self.commands_history = []
