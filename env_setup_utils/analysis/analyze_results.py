#!/usr/bin/env python3

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
from typing import Counter as CounterType

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import wandb


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line: {e}")
    return results


def extract_metrics(
    results: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Extract relevant metrics from results and collect all diagnostics."""
    metrics = []
    all_diagnostics = []

    for result in results:
        if not result.get("pyright"):
            continue

        metric = {
            "repo_name": result["repo_name"],
            "commit_sha": result["commit_sha"],
            "execution_time": result["execution_time"],
            "exit_code": result["exit_code"],
            "issues_count": result["issues_count"],
            "container_logs": result.get("container_logs", ""),
        }

        # Extract Pyright summary and diagnostics
        pyright_data = result["pyright"]
        summary = pyright_data.get("summary", {})
        metric.update(
            {
                "files_analyzed": summary.get("filesAnalyzed", 0),
                "error_count": summary.get("errorCount", 0),
                "warning_count": summary.get("warningCount", 0),
                "information_count": summary.get("informationCount", 0),
                "analysis_time": summary.get("timeInSec", 0),
            }
        )

        # Collect diagnostics
        if "generalDiagnostics" in pyright_data:
            all_diagnostics.extend(pyright_data["generalDiagnostics"])

        metrics.append(metric)

    return pd.DataFrame(metrics), all_diagnostics


def analyze_success_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze success rates and failures."""
    total_repos = len(df)

    # Successful repositories (exit_code=0 and issues_count=0)
    successful = df[(df["exit_code"] == 0) & (df["issues_count"] == 0)]
    success_count = len(successful)
    success_rate = (success_count / total_repos) * 100

    # Failed repositories (exit_code != 0)
    failed = df[df["exit_code"] != 0]
    failed_count = len(failed)
    failed_rate = (failed_count / total_repos) * 100

    # Repositories with issues (exit_code=0 but issues_count>0)
    with_issues = df[(df["exit_code"] == 0) & (df["issues_count"] > 0)]
    issues_count = len(with_issues)
    issues_rate = (issues_count / total_repos) * 100

    return {
        "total_repositories": total_repos,
        "successful_count": success_count,
        "successful_rate": success_rate,
        "failed_count": failed_count,
        "failed_rate": failed_rate,
        "with_issues_count": issues_count,
        "with_issues_rate": issues_rate,
        "avg_issues_per_repo": df[df["exit_code"] == 0]["issues_count"].mean(),
    }


def create_visualizations(df: pd.DataFrame, diagnostics: List[Dict[str, Any]], output_dir: Path):
    """Create interactive Plotly visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distribution of issues per repository (for successful runs)
    successful_df = df[df["exit_code"] == 0]
    fig_issues = px.histogram(
        successful_df,
        x="issues_count",
        title="Distribution of Issues per Repository (Successful Runs)",
        labels={"issues_count": "Number of Issues", "count": "Number of Repositories"},
        nbins=50,
    )
    fig_issues.write_html(output_dir / "issues_distribution.html")

    # 2. Execution time vs number of files
    fig_exec = px.scatter(
        df,
        x="files_analyzed",
        y="execution_time",
        color="exit_code",
        title="Execution Time vs Number of Files",
        labels={
            "files_analyzed": "Number of Files Analyzed",
            "execution_time": "Execution Time (s)",
            "exit_code": "Exit Code",
        },
    )
    fig_exec.write_html(output_dir / "execution_time_vs_files.html")

    # 3. Analysis of error types and their frequencies
    error_counts: CounterType[str] = Counter()
    warning_counts: CounterType[str] = Counter()

    for diag in diagnostics:
        rule = diag.get("rule", "no_rule")
        # Split by colon and take only the rule name
        rule = rule.split(":")[0] if ":" in rule else rule

        if diag.get("severity") == "error":
            error_counts[rule] += 1
        elif diag.get("severity") == "warning":
            warning_counts[rule] += 1

    # Create a combined bar plot for top errors and warnings
    top_errors = dict(error_counts.most_common(20))
    top_warnings = dict(warning_counts.most_common(20))

    fig_issues = go.Figure()

    # Add error bars
    fig_issues.add_trace(
        go.Bar(
            x=list(top_errors.keys()),
            y=list(top_errors.values()),
            name="Errors",
            marker_color="red",
        )
    )

    # Add warning bars
    fig_issues.add_trace(
        go.Bar(
            x=list(top_warnings.keys()),
            y=list(top_warnings.values()),
            name="Warnings",
            marker_color="yellow",
        )
    )

    fig_issues.update_layout(
        title="Top 20 Most Common Issues by Type",
        xaxis_title="Issue Type",
        yaxis_title="Frequency",
        barmode="group",
        showlegend=True,
        xaxis_tickangle=45,
    )

    fig_issues.write_html(output_dir / "issue_types.html")


def log_to_wandb(
    df: pd.DataFrame,
    success_metrics: Dict[str, Any],
    project_name: str,
    run_name: str,
    plots_dir: Path,
):
    """Log metrics and visualizations to Weights & Biases."""
    run = wandb.init(project=project_name, name=run_name)

    # Log success metrics
    wandb.log(success_metrics)

    # Log additional metrics
    wandb.log(
        {
            "total_files_analyzed": df["files_analyzed"].sum(),
            "avg_execution_time": df[df["exit_code"] == 0]["execution_time"].mean(),
            "total_errors": df["error_count"].sum(),
            "total_warnings": df["warning_count"].sum(),
        }
    )

    # Log visualizations
    for plot_path in plots_dir.glob("*.html"):
        wandb.log({plot_path.stem: wandb.Html(str(plot_path))})

    # Create and log tables
    # Main results table
    results_table = wandb.Table(
        dataframe=df[
            [
                "repo_name",
                "commit_sha",
                "exit_code",
                "issues_count",
                "files_analyzed",
                "execution_time",
            ]
        ]
    )
    wandb.log({"results_summary": results_table})

    # Failed repositories table with logs
    failed_df = df[df["exit_code"] != 0][["repo_name", "commit_sha", "exit_code", "container_logs"]]
    failed_table = wandb.Table(dataframe=failed_df)
    wandb.log({"failed_repositories": failed_table})

    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results and upload to W&B")
    parser.add_argument("results_file", type=str, help="Path to the JSONL results file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Directory to save analysis outputs",
    )
    parser.add_argument("--wandb-project", type=str, default="python-eval", help="W&B project name")
    parser.add_argument("--wandb-run", type=str, default="baseline-analysis", help="W&B run name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    # Load and process results
    logging.info(f"Loading results from {args.results_file}")
    results = load_jsonl(args.results_file)

    logging.info("Extracting metrics")
    df, diagnostics = extract_metrics(results)

    # Analyze success rates
    success_metrics = analyze_success_rates(df)
    logging.info(f"Success metrics: {json.dumps(success_metrics, indent=2)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processed data
    df.to_csv(output_dir / "processed_results.csv", index=False)

    # Create visualizations
    logging.info("Creating visualizations")
    create_visualizations(df, diagnostics, output_dir)

    # Upload to W&B
    logging.info("Uploading to Weights & Biases")
    log_to_wandb(df, success_metrics, args.wandb_project, args.wandb_run, output_dir)

    logging.info("Analysis complete!")


if __name__ == "__main__":
    main()
