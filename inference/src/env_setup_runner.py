import logging
import os
from datetime import datetime

import jsonlines
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from .agents.base import BaseEnvSetupAgent
from .agents.installamatic.agent import InstallamaticAgent


class EnvSetupRunner:
    def __init__(
        self,
        repository: str,
        revision: str,
        agent: BaseEnvSetupAgent,
        log_trajectory: bool,
        logging_dir: str,
    ):
        self.repository = repository
        self.revision = revision
        self.agent = agent

        self.log_trajectory = log_trajectory
        self.trajectory_file = os.path.join(logging_dir, f"{repository.replace('/', '__')}@{revision}.jsonl")
        os.makedirs(logging_dir, exist_ok=True)
        open(self.trajectory_file, "w").close()

    async def arun(self) -> None:
        initial_state = self.agent.construct_initial_state(repository=self.repository, revision=self.revision)

        graph = self.agent.get_agent()

        graph_config: RunnableConfig = {"configurable": self.agent.configurable_config}

        max_iterations = self.agent.max_iterations
        if max_iterations is not None:
            graph_config["recursion_limit"] = max_iterations

        try:
            async for current_update in graph.astream(
                initial_state,
                graph_config,
                stream_mode="updates",
                subgraphs=True,
            ):
                if isinstance(current_update, tuple):
                    current_update = current_update[1]

                current_update["timestamp"] = datetime.now().isoformat()

                if self.log_trajectory:
                    with jsonlines.open(self.trajectory_file, "a") as writer:
                        writer.write(self.agent.process_update_for_trajectory(current_update))
        except GraphRecursionError:
            logging.info("Agent stopped due to max iterations.")
        except Exception as e:
            logging.warning(f"Agent stopped due to an exception: {str(e)}.")

        if self.log_trajectory:
            if not isinstance(self.agent, InstallamaticAgent):
                with jsonlines.open(self.trajectory_file, "a") as writer:
                    writer.write(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "node": "commands_history",
                            "commands": self.agent.commands_history,
                        }
                    )

        return None
