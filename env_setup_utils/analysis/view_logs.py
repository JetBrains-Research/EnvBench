#!/usr/bin/env python3

import argparse
import json
import os
import re
import webbrowser
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from typing import Counter as CounterType

import plotly.graph_objects as go
from flask import Flask, redirect, render_template_string, request, url_for

from analysis.utils import get_file_path

app = Flask(__name__)

# Global variable to store the data
RESULTS_DATA: List[Dict[str, Any]] = []
BASELINE_DATA: List[Dict[str, Any]] = []

# Add exit code mapping
EXIT_CODE_MAP = {
    -127: "TIMEOUT",
    -999: "UNKNOWN_FAILURE",
    -888: "DOCKER_FAILURE",
    -777: "CREATE_CONTAINER_FAILURE",
    -666: "DOWNLOAD_FAILURE",
    -555: "SCRIPT_FAILURE",
}

# Add dependency manager mapping
DEPENDENCY_MANAGER_MAP = {
    # Python dependency managers
    "found requirements": "requirements.txt",
    "setup.py": "setup.py",
    "pyproject.toml": "pyproject",
    "setup.cfg": "setup.cfg",
    "installing from pipfile": "Pipfile",
    "detected conda environment": "Conda",
    "uv.lock": "uv",
    "poetry.lock": "Poetry",
    # JVM dependency managers
    "detected maven project": "Maven",
    "detected gradle project": "Gradle",
}


def analyze_issues(
    diagnostics: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, int], str]:
    """Analyze issue types and create a bar chart."""
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

    fig = go.Figure()

    # Add error bars with different colors for reportMissingImports
    error_x = list(top_errors.keys())
    error_y = list(top_errors.values())
    error_colors = ["#ff0000" if x == "reportMissingImports" else "#ff69b4" for x in error_x]

    fig.add_trace(
        go.Bar(
            x=error_x,
            y=error_y,
            name="Errors",
            marker_color=error_colors,
            showlegend=False,
        )
    )

    # Add warning bars
    fig.add_trace(
        go.Bar(
            x=list(top_warnings.keys()),
            y=list(top_warnings.values()),
            name="Warnings",
            marker_color="yellow",
        )
    )

    # Add custom legend entries for error types
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="Missing Import Errors",
            marker_color="#ff0000",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="Other Errors",
            marker_color="#ff69b4",
            showlegend=True,
        )
    )

    fig.update_layout(
        title="Top 20 Most Common Issues by Type",
        xaxis_title="Issue Type",
        yaxis_title="Frequency",
        barmode="group",
        showlegend=True,
        xaxis_tickangle=45,
        height=400,
        margin=dict(b=100),
    )

    return (
        dict(error_counts),
        dict(warning_counts),
        fig.to_html(full_html=False, include_plotlyjs=True),
    )


def calculate_stats(
    data: List[Dict[str, Any]], comparison_data: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Dict[str, Any], str, str]:
    """Calculate statistics and create charts."""
    total = len(data)

    # Create a mapping of repo+commit to data for comparison
    comparison_map = {}
    if comparison_data:
        comparison_map = {f"{r['repo_name']}@{r['commit_sha']}": r for r in comparison_data}

    # Find common repositories
    common_repos = set()
    if comparison_data:
        current_map = {f"{r['repo_name']}@{r['commit_sha']}": r for r in data}
        common_repos = set(current_map.keys()) & set(comparison_map.keys())

    # Determine mode based on first non-empty result
    is_python_mode = None
    for result in data:
        if result.get("pyright") is not None:
            is_python_mode = True
            break
        elif result.get("build_tool") is not None:
            is_python_mode = False
            break

    successful = 0
    with_issues = 0
    failed = 0
    total_missing_imports = 0
    total_missing_packages = 0
    non_error_repos = 0

    # For comparison
    prev_successful = 0
    prev_with_issues = 0
    prev_failed = 0
    prev_total_missing_imports = 0
    prev_total_missing_packages = 0
    prev_non_error_repos = 0

    for r in data:
        if r["exit_code"] == 0:  # Only consider non-error repositories
            non_error_repos += 1
            if is_python_mode:
                if r.get("issues_count", 0) == 0:
                    successful += 1
                else:
                    with_issues += 1
                if r.get("issues_count"):
                    total_missing_imports += r["issues_count"]
                if r.get("missing_packages_count"):
                    total_missing_packages += r["missing_packages_count"]
            else:  # JVM mode
                diagnostic_count = len(r.get("diagnostic_log", []))
                if diagnostic_count == 0:
                    successful += 1
                else:
                    with_issues += 1
                total_missing_imports += diagnostic_count
        else:
            failed += 1

        # Calculate comparison stats for common repos
        if comparison_data:
            key = f"{r['repo_name']}@{r['commit_sha']}"
            if key in comparison_map:
                prev_r = comparison_map[key]
                if prev_r["exit_code"] == 0:
                    prev_non_error_repos += 1
                    if is_python_mode:
                        if prev_r.get("issues_count", 0) == 0:
                            prev_successful += 1
                        else:
                            prev_with_issues += 1
                        if prev_r.get("issues_count"):
                            prev_total_missing_imports += prev_r["issues_count"]
                        if prev_r.get("missing_packages_count"):
                            prev_total_missing_packages += prev_r["missing_packages_count"]
                    else:  # JVM mode
                        diagnostic_count = len(prev_r.get("diagnostic_log", []))
                        if diagnostic_count == 0:
                            prev_successful += 1
                        else:
                            prev_with_issues += 1
                        prev_total_missing_imports += diagnostic_count
                else:
                    prev_failed += 1

    avg_missing_imports = total_missing_imports / non_error_repos if non_error_repos > 0 else 0
    avg_missing_packages = total_missing_packages / non_error_repos if non_error_repos > 0 else 0

    # Calculate comparison averages
    prev_avg_missing_imports = prev_total_missing_imports / prev_non_error_repos if prev_non_error_repos > 0 else 0
    prev_avg_missing_packages = prev_total_missing_packages / prev_non_error_repos if prev_non_error_repos > 0 else 0

    stats = {
        "total": total,
        "successful": successful,
        "with_issues": with_issues,
        "failed": failed,
        "total_missing_imports": total_missing_imports,
        "total_missing_packages": total_missing_packages,
        "avg_missing_imports": round(avg_missing_imports, 2),
        "avg_missing_packages": round(avg_missing_packages, 2),
    }

    if comparison_data:
        stats.update(
            {
                "common_repos": len(common_repos),
                "prev_successful": prev_successful,
                "prev_with_issues": prev_with_issues,
                "prev_failed": prev_failed,
                "prev_total_missing_imports": prev_total_missing_imports,
                "prev_total_missing_packages": prev_total_missing_packages,
                "prev_avg_missing_imports": round(prev_avg_missing_imports, 2),
                "prev_avg_missing_packages": round(prev_avg_missing_packages, 2),
                "delta_successful": successful - prev_successful,
                "delta_with_issues": with_issues - prev_with_issues,
                "delta_failed": failed - prev_failed,
                "delta_missing_imports": total_missing_imports - prev_total_missing_imports,
                "delta_missing_packages": total_missing_packages - prev_total_missing_packages,
                "delta_avg_missing_imports": round(avg_missing_imports - prev_avg_missing_imports, 2),
                "delta_avg_missing_packages": round(avg_missing_packages - prev_avg_missing_packages, 2),
            }
        )

    # Create status pie chart with transitions if comparison data exists
    pie_data = []
    if comparison_data:
        # Add arrows to show transitions
        pie_data.append(
            go.Pie(
                labels=["Successful", "With Issues", "Failed"],
                values=[prev_successful, prev_with_issues, prev_failed],
                hole=0.6,
                name="Previous",
                domain={"x": [0, 0.45]},
                marker=dict(colors=["#4caf50", "#ff9800", "#f44336"]),
                textinfo="value",
                hovertemplate="Previous: %{label}<br>Count: %{value}<extra></extra>",
            )
        )
        pie_data.append(
            go.Pie(
                labels=["Successful", "With Issues", "Failed"],
                values=[successful, with_issues, failed],
                hole=0.6,
                name="Current",
                domain={"x": [0.55, 1]},
                marker=dict(colors=["#4caf50", "#ff9800", "#f44336"]),
                textinfo="value",
                hovertemplate="Current: %{label}<br>Count: %{value}<extra></extra>",
            )
        )
    else:
        pie_data.append(
            go.Pie(
                labels=["Successful", "With Issues", "Failed"],
                values=[successful, with_issues, failed],
                hole=0.3,
                marker=dict(colors=["#4caf50", "#ff9800", "#f44336"]),
            )
        )

    pie_fig = go.Figure(data=pie_data)

    if comparison_data:
        pie_fig.update_layout(
            showlegend=True,
            margin=dict(t=0, b=0, l=0, r=0),
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(x=0.225, y=0.5, text="Previous", showarrow=False, font=dict(size=12)),
                dict(x=0.775, y=0.5, text="Current", showarrow=False, font=dict(size=12)),
            ],
        )
    else:
        pie_fig.update_layout(
            showlegend=True,
            margin=dict(t=0, b=0, l=0, r=0),
            height=200,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

    # Analyze all issues
    issues_chart = None
    if is_python_mode:
        all_diagnostics = []
        for result in data:
            if result.get("pyright") and result["pyright"].get("generalDiagnostics"):
                all_diagnostics.extend(result["pyright"]["generalDiagnostics"])
        _, _, issues_chart = analyze_issues(all_diagnostics)

    return stats, pie_fig.to_html(full_html=False, include_plotlyjs=True), issues_chart


def get_exit_code_display(code: int) -> str:
    """Convert exit code to display string."""
    return EXIT_CODE_MAP.get(code, str(code))


def extract_missing_packages(diagnostics: List[Dict[str, Any]]) -> set:
    """Extract unique missing packages from diagnostics messages."""
    missing_packages = set()
    pattern = r'Import "([^."]+)'

    for diag in diagnostics:
        if diag.get("rule") == "reportMissingImports":
            if match := re.search(pattern, diag.get("message", "")):
                missing_packages.add(match.group(1))

    return missing_packages


def get_github_url(repo_name: str, commit_sha: str, file_path: str, start_line: int, end_line: int) -> str:
    """Generate GitHub URL for the given file and line range."""
    # Remove /data/project/ prefix if present
    file_path = file_path.replace("/data/project/", "")
    # Convert repo name to GitHub format (assuming it's in org/repo format)
    # Add 1 to line numbers for 1-based indexing in GitHub
    return f"https://github.com/{repo_name}/blob/{commit_sha}/{file_path}#L{start_line + 1}-L{end_line + 1}"


def get_github_repo_url(repo_name: str, commit_sha: str) -> str:
    """Generate GitHub URL for the repository at a specific revision."""
    return f"https://github.com/{repo_name}/tree/{commit_sha}"


def detect_dependency_managers(container_logs: str) -> set:
    """Detect dependency managers used based on container logs."""
    managers: set[str] = set()
    if not container_logs:  # Handle None or empty string
        return managers

    logs_lower = container_logs.lower()
    # Only check for bootstrap script after "Running bootstrap script" appears
    if "running bootstrap script" in logs_lower:
        logs_after_bootstrap = logs_lower[logs_lower.index("running bootstrap script") :]
    else:
        logs_after_bootstrap = logs_lower

    for keyword, display_name in DEPENDENCY_MANAGER_MAP.items():
        if keyword.lower() in logs_after_bootstrap:
            managers.add(display_name)

    return managers


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                # Determine mode based on presence of pyright or build_tool
                is_python_mode = data.get("pyright") is not None

                if is_python_mode and data["exit_code"] == 0:
                    # Calculate missing packages if pyright diagnostics exist
                    diagnostics = data["pyright"].get("generalDiagnostics", [])
                    missing_packages = extract_missing_packages(diagnostics)
                    data["missing_packages"] = missing_packages
                    data["missing_packages_count"] = len(missing_packages)
                else:
                    # For JVM mode, set missing packages to N/A
                    data["missing_packages"] = None
                    data["missing_packages_count"] = None

                # Detect dependency managers
                data["dependency_managers"] = detect_dependency_managers(data.get("container_logs", ""))

                results.append(data)
            except json.JSONDecodeError:
                continue
    return results


# Home page template
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .stats-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .stat-item {
            text-align: center;
            padding: 10px;
        }
        .stat-number {
            font-size: 24px;
            font-weight: bold;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
        .delta-positive {
            color: #4caf50;
        }
        .delta-negative {
            color: #f44336;
        }
        .delta {
            font-size: 14px;
            margin-left: 5px;
        }
        .repo-row {
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .repo-row:hover {
            background-color: #f8f9fa;
        }
        .search-container {
            position: sticky;
            top: 0;
            background-color: white;
            padding: 15px 0;
            z-index: 100;
            border-bottom: 1px solid #dee2e6;
        }
        .index-badge {
            font-family: monospace;
            background-color: #6c757d;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container-fluid py-3">
        <div class="search-container mb-4">
            <div class="row align-items-center">
                <div class="col">
                    <h4>Evaluation Results</h4>
                </div>
                <div class="col-auto">
                    <form class="d-flex" method="GET">
                        <input type="text" name="search" class="form-control me-2" 
                               placeholder="Search repository..." value="{{ search }}">
                        <button type="submit" class="btn btn-primary">Search</button>
                        {% if search %}
                            <a href="/" class="btn btn-secondary ms-2">Clear</a>
                        {% endif %}
                    </form>
                </div>
            </div>
        </div>

        <!-- Statistics Section -->
        <div class="stats-card">
            <div class="row">
                <div class="col-md-4">
                    <div class="row">
                        <div class="col-4">
                            <div class="stat-item">
                                <div class="stat-number text-primary">
                                    {{ stats.total }}
                                    {% if stats.get('common_repos') %}
                                        <small class="text-muted">({{ stats.common_repos }} common)</small>
                                    {% endif %}
                                </div>
                                <div class="stat-label">Total Repositories</div>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="stat-item">
                                <div class="stat-number text-success">
                                    {{ stats.successful }}
                                    {% if stats.get('prev_successful') is not none %}
                                        <span class="delta {% if stats.delta_successful > 0 %}delta-positive{% elif stats.delta_successful < 0 %}delta-negative{% endif %}">
                                            ({{ '+' if stats.delta_successful > 0 else '' }}{{ stats.delta_successful }})
                                        </span>
                                    {% endif %}
                                </div>
                                <div class="stat-label">Successful</div>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="stat-item">
                                <div class="stat-number text-danger">
                                    {{ stats.failed }}
                                    {% if stats.get('prev_failed') is not none %}
                                        <span class="delta {% if stats.delta_failed < 0 %}delta-positive{% elif stats.delta_failed > 0 %}delta-negative{% endif %}">
                                            ({{ '+' if stats.delta_failed > 0 else '' }}{{ stats.delta_failed }})
                                        </span>
                                    {% endif %}
                                </div>
                                <div class="stat-label">Failed</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-8">
                    {{ pie_chart | safe }}
                </div>
            </div>
        </div>

        <!-- Missing Imports and Packages Stats -->
        <div class="stats-card mb-4">
            <div class="row">
                <div class="col-md-6">
                    <h5 class="mb-3">Missing Imports Statistics</h5>
                    <div class="row">
                        <div class="col-6">
                            <div class="stat-item">
                                <div class="stat-number text-info">
                                    {{ stats.total_missing_imports }}
                                    {% if stats.get('prev_total_missing_imports') is not none %}
                                        <span class="delta {% if stats.delta_missing_imports < 0 %}delta-positive{% elif stats.delta_missing_imports > 0 %}delta-negative{% endif %}">
                                            ({{ '+' if stats.delta_missing_imports > 0 else '' }}{{ stats.delta_missing_imports }})
                                        </span>
                                    {% endif %}
                                </div>
                                <div class="stat-label">Total {% if repos[0].get('pyright') is not none %}Missing Imports{% else %}Errors{% endif %}</div>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="stat-item">
                                <div class="stat-number text-info">
                                    {{ stats.avg_missing_imports }}
                                    {% if stats.get('prev_avg_missing_imports') is not none %}
                                        <span class="delta {% if stats.delta_avg_missing_imports < 0 %}delta-positive{% elif stats.delta_avg_missing_imports > 0 %}delta-negative{% endif %}">
                                            ({{ '+' if stats.delta_avg_missing_imports > 0 else '' }}{{ stats.delta_avg_missing_imports }})
                                        </span>
                                    {% endif %}
                                </div>
                                <div class="stat-label">Avg {% if repos[0].get('pyright') is not none %}Missing Imports{% else %}Errors{% endif %} per Repo</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    {% if repos[0].get('pyright') is not none %}
                    <h5 class="mb-3">Missing Packages Statistics</h5>
                    <div class="row">
                        <div class="col-6">
                            <div class="stat-item">
                                <div class="stat-number text-warning">
                                    {{ stats.total_missing_packages }}
                                    {% if stats.get('prev_total_missing_packages') is not none %}
                                        <span class="delta {% if stats.delta_missing_packages < 0 %}delta-positive{% elif stats.delta_missing_packages > 0 %}delta-negative{% endif %}">
                                            ({{ '+' if stats.delta_missing_packages > 0 else '' }}{{ stats.delta_missing_packages }})
                                        </span>
                                    {% endif %}
                                </div>
                                <div class="stat-label">Total Missing Packages</div>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="stat-item">
                                <div class="stat-number text-warning">
                                    {{ stats.avg_missing_packages }}
                                    {% if stats.get('prev_avg_missing_packages') is not none %}
                                        <span class="delta {% if stats.delta_avg_missing_packages < 0 %}delta-positive{% elif stats.delta_avg_missing_packages > 0 %}delta-negative{% endif %}">
                                            ({{ '+' if stats.delta_avg_missing_packages > 0 else '' }}{{ stats.delta_avg_missing_packages }})
                                        </span>
                                    {% endif %}
                                </div>
                                <div class="stat-label">Avg Missing Packages per Repo</div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>

        <!-- Issue Distribution -->
        {% if issues_chart %}
        <div class="card mb-4">
            <div class="card-body">
                {{ issues_chart | safe }}
            </div>
        </div>
        {% endif %}

        <!-- Results Table -->
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Index</th>
                        <th style="max-width: 200px; overflow: hidden; text-overflow: ellipsis;">Repository</th>
                        <th>Commit</th>
                        <th>Status</th>
                        <th>Errors</th>
                        {% if repos[0].get('pyright') is not none %}
                            <th>Missing Packages</th>
                        {% else %}
                            <th>Build Tool</th>
                        {% endif %}
                        <th>Dep Managers</th>
                        <th>Execution Time</th>
                    </tr>
                </thead>
                <tbody>
                    {% for repo in repos %}
                    {% set baseline = baseline_map.get(repo.repo_name + '@' + repo.commit_sha) if baseline_map else None %}
                    <tr class="repo-row" onclick="window.location='/logs/{{ loop.index0 }}'">
                        <td><span class="index-badge">{{ loop.index0 }}</span></td>
                        <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{{ repo.repo_name }}">
                            {{ repo.repo_name }}
                        </td>
                        <td><code>{{ repo.commit_sha[:8] }}</code></td>
                        <td>
                            <span class="badge {% if repo.exit_code == 0 %}bg-success{% else %}bg-danger{% endif %}"
                                  title="{% if repo.error %}{{ repo.error }}{% endif %}">
                                {% if repo.exit_code == 0 %}Success{% else %}{{ get_exit_code_display(repo.exit_code) }}{% endif %}
                            </span>
                            {% if baseline %}
                                <span class="badge {% if baseline.exit_code == 0 %}bg-success{% else %}bg-danger{% endif %} opacity-50"
                                      title="Baseline status">
                                    {% if baseline.exit_code == 0 %}Success{% else %}{{ get_exit_code_display(baseline.exit_code) }}{% endif %}
                                </span>
                            {% endif %}
                        </td>
                        <td>
                            {% if repo.exit_code == 0 %}
                                {% if repo.get('pyright') is not none %}
                                    <span class="badge {% if repo.issues_count == 0 %}bg-success{% else %}bg-warning{% endif %}"
                                          title="Number of missing import errors">
                                        {{ repo.issues_count }}
                                    </span>
                                    {% if baseline and baseline.exit_code == 0 %}
                                        <span class="badge {% if baseline.issues_count == 0 %}bg-success{% else %}bg-warning{% endif %} opacity-50"
                                              title="Baseline missing imports">
                                            {{ baseline.issues_count }}
                                        </span>
                                    {% endif %}
                                {% else %}
                                    <span class="badge {% if not repo.get('diagnostic_log') %}bg-success{% else %}bg-warning{% endif %}"
                                          title="Number of errors">
                                        {{ repo.get('diagnostic_log', [])|length }}
                                    </span>
                                    {% if baseline and baseline.exit_code == 0 %}
                                        <span class="badge {% if not baseline.get('diagnostic_log') %}bg-success{% else %}bg-warning{% endif %} opacity-50"
                                              title="Baseline number of errors">
                                            {{ baseline.get('diagnostic_log', [])|length }}
                                        </span>
                                    {% endif %}
                                {% endif %}
                            {% else %}
                                <span class="badge bg-secondary">N/A</span>
                            {% endif %}
                        </td>
                        {% if repo.get('pyright') is not none %}
                            <td>
                                {% if repo.exit_code == 0 and repo.missing_packages_count %}
                                    <span class="badge bg-info">{{ repo.missing_packages_count }}</span>
                                    {% if baseline and baseline.exit_code == 0 and baseline.missing_packages_count %}
                                        <span class="badge bg-info opacity-50">{{ baseline.missing_packages_count }}</span>
                                    {% endif %}
                                {% else %}
                                    <span class="badge bg-secondary">N/A</span>
                                {% endif %}
                            </td>
                        {% else %}
                            <td>
                                {% if repo.build_tool %}
                                    <span class="badge bg-info">{{ repo.build_tool }}</span>
                                    {% if baseline and baseline.build_tool %}
                                        <span class="badge bg-info opacity-50">{{ baseline.build_tool }}</span>
                                    {% endif %}
                                {% else %}
                                    <span class="badge bg-secondary">N/A</span>
                                {% endif %}
                            </td>
                        {% endif %}
                        <td>
                            {% if repo.dependency_managers %}
                                {% if repo.dependency_managers|length == 1 %}
                                    <span class="badge bg-primary">
                                        {{ repo.dependency_managers|list|first }}
                                    </span>
                                {% else %}
                                    <span class="badge bg-primary" title="{{ ', '.join(repo.dependency_managers) }}">
                                        {{ repo.dependency_managers|length }}
                                    </span>
                                {% endif %}
                            {% else %}
                                <span class="badge bg-secondary" title="No dependency managers detected">0</span>
                            {% endif %}
                        </td>
                        <td>
                            {{ "%.2f"|format(repo.execution_time) }}s
                            {% if baseline %}
                                <small class="text-muted">({{ "%.2f"|format(baseline.execution_time) }}s)</small>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

# Logs page template
LOGS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Script View</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .navbar {
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            padding: 0.5rem 1rem;
        }
        .script-container {
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
            margin: 20px 0;
        }
        .index-badge {
            font-family: monospace;
            background-color: #6c757d;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <nav class="navbar sticky-top">
        <div class="container-fluid">
            <a href="/" class="btn btn-outline-primary">
                <i class="bi bi-house-door"></i> Home
            </a>
        </div>
    </nav>
    <div class="container py-4">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <div>
                <h4>
                    <span class="index-badge">{{ index }}</span>
                    <span class="text-primary">{{ repo.repo_name }}</span>
                </h4>
                <p class="mb-0">
                    Revision: <code>{{ repo.commit_sha }}</code>
                    <a href="{{ get_github_repo_url(repo.repo_name, repo.commit_sha) }}" 
                       class="btn btn-sm btn-outline-secondary ms-2" target="_blank">
                        View on GitHub
                    </a>
                </p>
                {% if repo.exit_code != 0 %}
                    <p class="text-danger mt-2">
                        Status: {{ get_exit_code_display(repo.exit_code) }}
                        {% if repo.error %}
                            <br>Error: {{ repo.error }}
                        {% endif %}
                    </p>
                {% endif %}
                {% if baseline_repo %}
                    <div class="baseline-info mt-3 p-3 bg-light rounded">
                        <h6 class="mb-2">Baseline Information</h6>
                        <p class="mb-1">
                            Status: 
                            <span class="badge {% if baseline_repo.exit_code == 0 %}bg-success{% else %}bg-danger{% endif %}">
                                {% if baseline_repo.exit_code == 0 %}Success{% else %}{{ get_exit_code_display(baseline_repo.exit_code) }}{% endif %}
                            </span>
                        </p>
                        {% if baseline_repo.exit_code != 0 and baseline_repo.error %}
                            <p class="text-danger mb-1">Error: {{ baseline_repo.error }}</p>
                        {% endif %}
                        {% if repo.get('pyright') is not none %}
                            {% if baseline_repo.exit_code == 0 %}
                                <p class="mb-1">
                                    Missing Imports: 
                                    <span class="badge {% if baseline_repo.issues_count == 0 %}bg-success{% else %}bg-warning{% endif %}">
                                        {{ baseline_repo.issues_count }}
                                    </span>
                                </p>
                                {% if baseline_repo.missing_packages_count %}
                                    <p class="mb-1">
                                        Missing Packages: 
                                        <span class="badge bg-info">
                                            {{ baseline_repo.missing_packages_count }}
                                        </span>
                                    </p>
                                {% endif %}
                            {% endif %}
                        {% else %}
                            {% if baseline_repo.build_tool %}
                                <p class="mb-1">
                                    Build Tool: 
                                    <span class="badge bg-info">{{ baseline_repo.build_tool }}</span>
                                </p>
                            {% endif %}
                            {% if baseline_repo.exit_code == 0 %}
                                <p class="mb-1">
                                    Issues: 
                                    <span class="badge {% if not baseline_repo.get('diagnostic_log') %}bg-success{% else %}bg-warning{% endif %}">
                                        {{ baseline_repo.get('diagnostic_log', [])|length }}
                                    </span>
                                </p>
                            {% endif %}
                        {% endif %}
                        <p class="mb-0">
                            Execution Time: {{ "%.2f"|format(baseline_repo.execution_time) }}s
                        </p>
                    </div>
                {% endif %}
                {% if repo.dependency_managers %}
                    <p class="text-primary mt-2">
                        Dependency Managers: {{ ', '.join(repo.dependency_managers) }}
                    </p>
                {% else %}
                    <p class="text-muted mt-2">
                        No dependency managers detected
                    </p>
                {% endif %}
                {% if repo.get('pyright') is not none %}
                    {% if repo.exit_code == 0 and repo.missing_packages %}
                        <p class="text-info mt-2">
                            Missing Packages: {{ ', '.join(repo.missing_packages) }}
                        </p>
                    {% endif %}
                {% else %}
                    {% if repo.build_tool %}
                        <p class="text-info mt-2">
                            Build Tool: {{ repo.build_tool }}
                        </p>
                    {% endif %}
                {% endif %}
            </div>
            <div class="btn-group">
                {% if prev_idx is not none %}
                    <a href="/logs/{{ prev_idx }}" class="btn btn-outline-primary">Previous</a>
                {% endif %}
                {% if next_idx is not none %}
                    <a href="/logs/{{ next_idx }}" class="btn btn-outline-primary">Next</a>
                {% endif %}
            </div>
        </div>
        
        <div class="script-container">{{ repo.container_logs }}</div>
        
        {% if repo.exit_code == 0 %}
            {% if repo.get('pyright') is not none %}
                <div class="diagnostics-section">
                    <div class="diagnostics-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Pyright Diagnostics</h5>
                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="showImportsOnly" checked>
                            <label class="form-check-label" for="showImportsOnly">Show only missing imports</label>
                        </div>
                    </div>
                    <div class="diagnostics-body">
                        {% if repo['pyright'].get('generalDiagnostics') %}
                            <ul class="diagnostic-list">
                                {% for diag in repo['pyright']['generalDiagnostics'] %}
                                    <li class="diagnostic-item diagnostic-row {% if diag.get('rule') == 'reportMissingImports' %}import-error{% else %}other-error{% endif %}"
                                        {% if diag.get('rule') != 'reportMissingImports' %}style="display: none;"{% endif %}>
                                        <span class="diagnostic-location">
                                            <a href="{{ github_url(repo.repo_name, repo.commit_sha, diag.file, diag.range.start.line, diag.range.end.line) }}"
                                               class="github-link" target="_blank">
                                                {{ diag.file.split('/')[-1] }}:{{ diag.range.start.line + 1 }}
                                            </a>
                                        </span>
                                        <span class="diagnostic-message {{ 'error-text' if diag.severity == 'error' else 'warning-text' }}">
                                            {{ diag.message }}
                                            {% if diag.get('rule') %}
                                                <small class="text-muted">({{ diag.rule }})</small>
                                            {% endif %}
                                        </span>
                                    </li>
                                {% endfor %}
                            </ul>
                        {% else %}
                            <p class="text-muted mb-0">No diagnostics available</p>
                        {% endif %}
                    </div>
                </div>
            {% else %}
                {% if repo.get('diagnostic_log') %}
                    <div class="diagnostics-section mt-4">
                        <h5 class="mb-3">Build Diagnostics</h5>
                        <ul class="list-group">
                            {% for diagnostic in repo.diagnostic_log %}
                                <li class="list-group-item">{{ diagnostic }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                {% endif %}
            {% endif %}
        {% endif %}
        
        {% if repo.exit_code == 0 and issues_chart %}
            <div class="card mt-4">
                <div class="card-body">
                    {{ issues_chart | safe }}
                </div>
            </div>
        {% endif %}
    </div>

    <script>
        // Navigation keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            {% if prev_idx is not none %}
                if (event.key === 'ArrowLeft') {
                    window.location.href = '/logs/{{ prev_idx }}';
                }
            {% endif %}
            {% if next_idx is not none %}
                if (event.key === 'ArrowRight') {
                    window.location.href = '/logs/{{ next_idx }}';
                }
            {% endif %}
            if (event.key === 'h') {
                window.location.href = '/';
            }
        });

        // Diagnostic filter toggle
        {% if repo.get('pyright') is not none %}
            document.getElementById('showImportsOnly').addEventListener('change', function(e) {
                const showOnlyImports = e.target.checked;
                document.querySelectorAll('.diagnostic-row').forEach(row => {
                    if (row.classList.contains('import-error')) {
                        row.style.display = '';
                    } else {
                        row.style.display = showOnlyImports ? 'none' : '';
                    }
                });
            });
        {% endif %}
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="View logs data in a web interface")
    parser.add_argument("logs_file", type=str, nargs="?", help="Path to the JSONL logs file")
    parser.add_argument("--baseline", type=str, help="Path to baseline JSONL logs file to compare against")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--repo", type=str, help="Hugging Face repository to download from")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache and force redownload")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    args = parser.parse_args()

    # Get the file path, downloading if necessary
    file_path = get_file_path(
        file_path=args.logs_file,
        caller_name="view_logs",
        repo_id=args.repo if args.repo else None,
        no_cache=args.no_cache,
    )

    # Load data
    global RESULTS_DATA, BASELINE_DATA
    RESULTS_DATA = load_jsonl(file_path)

    if args.baseline:
        baseline_path = get_file_path(
            file_path=args.baseline,
            caller_name="view_logs",
            repo_id=args.repo if args.repo else None,
            no_cache=args.no_cache,
        )
        BASELINE_DATA = load_jsonl(baseline_path)

    url = f"http://{args.host}:{args.port}"
    print(f"Loaded {len(RESULTS_DATA)} logs. Starting server at {url}")
    if args.baseline:
        print(f"Loaded {len(BASELINE_DATA)} baseline logs.")

    # Only open browser in the main process, not in the reloader
    if not args.no_browser and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        webbrowser.open(url)

    app.run(host=args.host, port=args.port, debug=True)


@app.route("/")
def index():
    search = request.args.get("search", "").lower()

    # Filter results if search term is provided
    filtered_results = RESULTS_DATA
    filtered_baseline = BASELINE_DATA
    if search:
        filtered_results = [
            r for r in RESULTS_DATA if search in r["repo_name"].lower() or search in r["commit_sha"].lower()
        ]
        if BASELINE_DATA:
            filtered_baseline = [
                r for r in BASELINE_DATA if search in r["repo_name"].lower() or search in r["commit_sha"].lower()
            ]

    # Calculate statistics
    stats, pie_chart, issues_chart = calculate_stats(filtered_results, filtered_baseline if BASELINE_DATA else None)

    # Create a mapping of repo+commit to baseline data for table comparison
    baseline_map = {}
    if BASELINE_DATA:
        baseline_map = {f"{r['repo_name']}@{r['commit_sha']}": r for r in BASELINE_DATA}

    return render_template_string(
        HOME_TEMPLATE,
        repos=filtered_results,
        stats=stats,
        pie_chart=pie_chart,
        issues_chart=issues_chart,
        search=search,
        get_exit_code_display=get_exit_code_display,
        baseline_map=baseline_map,
    )


@app.route("/logs/<int:idx>")
def view_logs(idx: int):
    if idx < 0 or idx >= len(RESULTS_DATA):
        return redirect(url_for("index"))

    repo = RESULTS_DATA[idx]
    prev_idx = idx - 1 if idx > 0 else None
    next_idx = idx + 1 if idx < len(RESULTS_DATA) - 1 else None

    # Get baseline data if available
    baseline_repo = None
    if BASELINE_DATA:
        key = f"{repo['repo_name']}@{repo['commit_sha']}"
        baseline_repo = next((r for r in BASELINE_DATA if f"{r['repo_name']}@{r['commit_sha']}" == key), None)

    # Calculate issue distribution for this repository
    issues_chart = None
    if repo["exit_code"] == 0 and repo.get("pyright"):
        diagnostics = repo["pyright"].get("generalDiagnostics", [])
        _, _, issues_chart = analyze_issues(diagnostics)

    return render_template_string(
        LOGS_TEMPLATE,
        repo=repo,
        baseline_repo=baseline_repo,
        index=idx,
        prev_idx=prev_idx,
        next_idx=next_idx,
        issues_chart=issues_chart,
        github_url=get_github_url,
        get_github_repo_url=get_github_repo_url,
        get_exit_code_display=get_exit_code_display,
    )


def generate_logs_html(results_data: List[Dict[str, Any]], baseline_data: Optional[List[Dict[str, Any]]] = None) -> str:
    """Generate HTML string for the logs data home page.

    Args:
        results_data: List of log entries
        baseline_data: Optional list of baseline log entries to compare against

    Returns:
        str: HTML string for the home page
    """
    # Filter results if search term is provided
    filtered_results = results_data
    filtered_baseline = baseline_data

    # Calculate statistics
    stats, pie_chart, issues_chart = calculate_stats(filtered_results, filtered_baseline if baseline_data else None)

    # Create a mapping of repo+commit to baseline data for table comparison
    baseline_map = {}
    if baseline_data:
        baseline_map = {f"{r['repo_name']}@{r['commit_sha']}": r for r in baseline_data}

    with app.app_context():
        return render_template_string(
            HOME_TEMPLATE,
            repos=filtered_results,
            stats=stats,
            pie_chart=pie_chart,
            issues_chart=issues_chart,
            search="",
            get_exit_code_display=get_exit_code_display,
            baseline_map=baseline_map,
        )


def generate_logs_html_from_hf(
    logs_file: str,
    repo_id: str,
    baseline_file: Optional[str] = None,
    no_cache: bool = False,
) -> str:
    """Generate HTML string for logs data loaded from Hugging Face.

    Args:
        logs_file: Path to the JSONL logs file in the HF repo
        repo_id: Hugging Face repository ID
        baseline_file: Optional path to baseline JSONL file in the same HF repo
        no_cache: Whether to bypass cache and force redownload

    Returns:
        str: HTML string for the home page
    """
    # Get the file path, downloading if necessary
    file_path = get_file_path(
        file_path=logs_file,
        caller_name="view_logs",
        repo_id=repo_id,
        no_cache=no_cache,
    )

    # Load results data
    results_data = load_jsonl(file_path)

    # Load baseline data if provided
    baseline_data = None
    if baseline_file:
        baseline_path = get_file_path(
            file_path=baseline_file,
            caller_name="view_logs",
            repo_id=repo_id,
            no_cache=no_cache,
        )
        baseline_data = load_jsonl(baseline_path)

    return generate_logs_html(results_data, baseline_data)


if __name__ == "__main__":
    main()
