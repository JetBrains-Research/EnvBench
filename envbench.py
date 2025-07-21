import asyncio
from collections import deque
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Callable, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from rich.align import Align
from rich.box import DOUBLE, ROUNDED
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

# Configure rich traceback for better error display
from rich.traceback import install
import wandb

from env_setup_utils.analysis.scripts_viewer import generate_scripts_html_from_hf
from env_setup_utils.analysis.traj_viewer import generate_trajectories_html_from_hf
from env_setup_utils.analysis.view_logs import generate_logs_html_from_hf
from env_setup_utils.process_trajectories_to_scripts import process_trajectories_to_scripts
from evaluation.main import main as run_evaluation

# Import inference and evaluation modules
from inference.main import main as run_inference

install(show_locals=True, width=120, word_wrap=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True, markup=True)])
logger = logging.getLogger("pipeline")

STEP_STYLES = {"inference": "blue", "processing": "green", "evaluation": "yellow"}

STEP_EMOJIS = {"inference": "🤖", "processing": "🔄", "evaluation": "📊"}

# Global fancy output setting
FANCY_OUTPUT = False


@dataclass
class PipelineConfig:
    tmp_dir: str
    file_name: str
    run_name: str
    tag: str
    wandb_project: Optional[str] = "envsetup-oss"
    use_wandb: bool = True
    skip_inference: bool = False
    skip_processing: bool = False
    skip_evaluation: bool = False
    data_path: str = str(Path.home() / "data_path")
    fancy_output: bool = True


cs = ConfigStore.instance()
cs.store(name="pipeline_config", node=PipelineConfig)


def create_fancy_header(text: str, style: str = "magenta") -> Panel:
    """Create a fancy header with a double-line box and centered text"""
    return Panel(
        Align.center(text),
        style=style,
        box=DOUBLE,
        padding=(1, 2),
        title="✨ Environment Setup Pipeline ✨",
        title_align="center",
    )


def create_step_header(step: str, number: int, style: str) -> Panel:
    """Create a fancy step header with emoji and rounded box"""
    emoji = STEP_EMOJIS.get(step.lower(), "🔧")
    return Panel(
        f"{emoji} Step {number}: {step}", style=style, box=ROUNDED, padding=(1, 2), title="⚡️", title_align="right"
    )


def stream_subprocess_output(
    process: subprocess.Popen, description: str, progress: Progress, style: str = "blue"
) -> None:
    """Stream subprocess output in a nicely formatted way"""
    global FANCY_OUTPUT
    output_queue = queue.Queue()
    output_lines = deque(maxlen=15)
    task_id = progress.add_task(description, total=None)
    last_update = time.time()
    update_interval = 0.5  # Update every 0.5 seconds

    def reader(pipe, queue):
        try:
            with pipe:
                for line in iter(pipe.readline, ""):
                    queue.put(line.strip())
        finally:
            queue.put(None)

    stdout_thread = threading.Thread(target=reader, args=(process.stdout, output_queue))
    stderr_thread = threading.Thread(target=reader, args=(process.stderr, output_queue))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    # Create initial panel if using fancy output
    if progress.console.is_interactive and FANCY_OUTPUT:
        panel = Panel(
            "",
            title=f"[bold]{description}[/bold]",
            border_style=style,
            box=ROUNDED,
            title_align="center",
            padding=(1, 2),
            subtitle="🔄 Live Output",
            subtitle_align="right",
        )
        progress.console.print(panel)

    while stdout_thread.is_alive() or stderr_thread.is_alive() or not output_queue.empty():
        try:
            line = output_queue.get(timeout=0.1)
            if line is None:
                continue
            output_lines.append(line)

            # Update output based on mode
            current_time = time.time()
            if current_time - last_update >= update_interval:
                if progress.console.is_interactive and FANCY_OUTPUT:
                    progress.console.print(
                        Panel(
                            "\n".join(output_lines),
                            title=f"[bold]{description}[/bold]",
                            border_style=style,
                            box=ROUNDED,
                            title_align="center",
                            padding=(1, 2),
                            subtitle="🔄 Live Output",
                            subtitle_align="right",
                        )
                    )
                else:
                    # Simple output mode
                    for line in output_lines:
                        progress.console.print(escape(line))
                    output_lines.clear()
                last_update = current_time

        except queue.Empty:
            continue

    # Print final state
    if output_lines:
        if progress.console.is_interactive and FANCY_OUTPUT:
            progress.console.print(
                Panel(
                    "\n".join(output_lines),
                    title=f"[bold]{description}[/bold]",
                    border_style=style,
                    box=ROUNDED,
                    title_align="center",
                    padding=(1, 2),
                    subtitle="✅ Complete",
                    subtitle_align="right",
                )
            )
        else:
            # Simple output mode
            for line in output_lines:
                progress.console.print(escape(line))

    process.wait()
    progress.remove_task(task_id)


def run_command_with_progress(
    func: Callable,
    args: tuple,
    description: str,
    progress: Progress,
    style: str = "blue",
    data_path: str | None = None,
    count_pattern: str | None = None,
) -> None:
    """Run a coroutine function with progress tracking"""
    progress.console.print(
        Panel(
            f"[bold]Running function:[/bold]\n{func.__name__}: {description}", style=style, box=ROUNDED, padding=(1, 2)
        )
    )
    if data_path:
        os.environ["DATA_ROOT"] = data_path

    task_id = progress.add_task(f"[{style}]{description}", total=None)

    try:
        res = func(*args)
        if asyncio.iscoroutine(res):
            asyncio.run(res)
    except Exception as e:
        progress.remove_task(task_id)
        raise e

    progress.remove_task(task_id)


def create_artifact_table(title: str, repo_id: str, artifacts: list[tuple[str, str]], style: str = "blue") -> Table:
    """Create a table showing HuggingFace artifacts"""
    table = Table(title=title, box=ROUNDED, style=style, title_style=f"bold {style}")
    table.add_column("Artifact", style="bold")
    table.add_column("Location on HuggingFace")

    for name, path in artifacts:
        table.add_row(name, f"{repo_id}/{path}")

    return table


def create_summary_table(artifacts: list[tuple[str, str, str]]) -> Table:
    """Create a summary table showing all HuggingFace artifacts"""
    table = Table(
        title="🗂️  Pipeline Artifacts Summary",
        box=ROUNDED,
        title_style="bold magenta",
        caption="All artifacts have been uploaded to HuggingFace 🤗",
        caption_style="dim",
    )
    table.add_column("Stage", style="bold")
    table.add_column("Repository", style="cyan")
    table.add_column("Path", style="green")

    for stage, repo, path in artifacts:
        table.add_row(stage, repo, path)

    return table


def resolve_cfg(cfg):
    return OmegaConf.create(OmegaConf.to_object(cfg))  # hack


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:
    console = Console()
    # Track all artifacts for final summary
    artifacts = []

    # Print fancy header
    console.print(create_fancy_header("[bold]Environment Setup Pipeline[/bold]"))

    # Print configuration panel
    config_text = [
        "[bold white]Configuration Details[/bold white]",
        "─" * 50,
        f"[bold]🏷️  Run name:[/bold] {cfg.run_name}",
        f"[bold]💾 Data path:[/bold] {cfg.data_path}",
        "[bold]🔄 Active Steps:[/bold] "
        + " ".join(
            [
                f"[{STEP_STYLES[step.lower()]}]{step}[/{STEP_STYLES[step.lower()]}]"
                for step, enabled in {
                    "Inference": not cfg.skip_inference,
                    "Processing": not cfg.skip_processing,
                    "Evaluation": not cfg.skip_evaluation,
                }.items()
                if enabled
            ]
        ),
        f"[bold]📊 Wandb:[/bold] {'[green]enabled[/green]' if cfg.use_wandb else '[red]disabled[/red]'}",
    ]
    console.print(Panel("\n".join(config_text), box=ROUNDED, style="cyan", padding=(1, 2)))
    console.print(Rule(style="bright_black"))

    # Load base configuration
    base_config: DictConfig = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="bright_green"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Step 1: Run Inference
        if not cfg.skip_inference:
            if cfg.use_wandb:
                wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    job_type="inference",
                    name=f"{cfg.run_name} inference",
                    config=base_config["inference"],
                )
            console.print(create_step_header("Inference", 1, "blue"))
            run_command_with_progress(
                run_inference,
                (resolve_cfg(cfg.inference),),
                "Running inference...",
                progress,
                style="blue",
                data_path=cfg.data_path,
            )

            console.print(Panel("🔍 Generating trajectories visualization...", style="blue", box=ROUNDED))
            traj_html = generate_trajectories_html_from_hf(
                traj_dir=f"{cfg.run_name}/trajectories",
                repo_id=base_config["inference"]["hf"]["repo_id"],
                no_cache=True,
            )
            if cfg.use_wandb:
                wandb.log({"trajectories_viewer": wandb.Html(traj_html)})
                wandb_run.finish()

            # Track artifact
            artifacts.append(("Inference", base_config["inference"]["hf"]["repo_id"], f"{cfg.run_name}/trajectories"))
            console.print(Rule(style="bright_black"))

        # Step 2: Process Trajectories
        if not cfg.skip_processing:
            if cfg.use_wandb:
                wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    job_type="scripts",
                    name=f"{cfg.run_name} scripts",
                )
            console.print(create_step_header("Processing", 2, "green"))
            run_command_with_progress(
                process_trajectories_to_scripts,
                (
                    base_config["inference"]["hf"]["repo_id"],
                    cfg.run_name,
                ),
                "Processing trajectories...",
                progress,
                style="green",
                data_path=cfg.data_path,
            )
            console.print(Panel("📝 Generating scripts visualization...", style="green", box=ROUNDED))
            scripts_html = generate_scripts_html_from_hf(
                scripts_file=f"{cfg.run_name}/scripts.jsonl",
                repo_id=base_config["inference"]["hf"]["repo_id"],
                no_cache=True,
            )
            if cfg.use_wandb:
                wandb.log({"scripts_viewer": wandb.Html(scripts_html)})
                wandb_run.finish()

            # Track artifact
            artifacts.append(("Processing", base_config["inference"]["hf"]["repo_id"], f"{cfg.run_name}/scripts.jsonl"))
            console.print(Rule(style="bright_black"))

        # Step 3: Evaluation
        if not cfg.skip_evaluation:
            if cfg.use_wandb:
                wandb_run = wandb.init(
                    project=str(cfg.wandb_project),
                    job_type="evaluation",
                    name=f"{cfg.run_name} eval",
                    config=base_config["evaluation"],
                )
            console.print(create_step_header("Evaluation", 3, "yellow"))
            run_command_with_progress(
                run_evaluation,
                (resolve_cfg(cfg.evaluation),),
                "Running evaluation...",
                progress,
                style="yellow",
                data_path=cfg.data_path,
                count_pattern=r"Found\s\d+\sissues",  # Add pattern for counting repositories
            )

            console.print(Panel("📊 Generating evaluation visualization...", style="yellow", box=ROUNDED))
            eval_html = generate_logs_html_from_hf(
                logs_file=f"{cfg.run_name}/results.jsonl",
                repo_id=base_config["evaluation"]["output"]["hf"]["repo_id"],
                no_cache=True,
            )
            if cfg.use_wandb:
                wandb.log({"evaluation_viewer": wandb.Html(eval_html)})
                wandb_run.finish()

            # Track artifact
            artifacts.append(
                ("Evaluation", base_config["evaluation"]["output"]["hf"]["repo_id"], f"{cfg.run_name}/results.jsonl")
            )
            console.print(Rule(style="bright_black"))

    # Print artifacts summary table
    if artifacts:
        console.print("\n")
        console.print(create_summary_table(artifacts))
        console.print("\n")

    # Print completion message
    console.print(
        Panel(
            Align.center("[bold]🎉 Pipeline completed successfully! 🎉[/bold]"),
            style="bold green",
            box=DOUBLE,
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    main()
