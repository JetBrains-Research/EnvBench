#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List
import webbrowser

from flask import Flask, redirect, render_template_string, url_for

from env_setup_utils.analysis.utils import get_dir_path

# Model pricing per 1M tokens (placeholder values)
MODEL_PRICING = {
    "gpt-4o": {"input": 2.5, "output": 10.0},  # $2.50 per 1M input tokens, $10.00 per 1M output tokens
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # $0.15 per 1M input tokens, $0.60 per 1M output tokens
}


def calculate_message_cost(message: Dict[str, Any]) -> Dict[str, float]:
    """Calculate the cost and token counts for a single message."""
    if (
        message.get("node") != "agent"
        or not message.get("messages")
        or not message["messages"][0].get("message_content", {}).get("usage_metadata")
    ):
        return {"cost": 0.0, "input_tokens": 0, "output_tokens": 0}

    metadata = message["messages"][0]["message_content"]["usage_metadata"]
    model_name = message["messages"][0].get("response_metadata", {}).get("model_name", "")

    # Determine pricing based on model
    if "gpt-4o-mini" in model_name:
        pricing = MODEL_PRICING["gpt-4o-mini"]
    else:
        pricing = MODEL_PRICING["gpt-4o"]

    input_tokens = metadata.get("input_tokens", 0)
    output_tokens = metadata.get("output_tokens", 0)

    # Calculate cost using per 1M token pricing
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    return {"cost": cost, "input_tokens": input_tokens, "output_tokens": output_tokens}


app = Flask(__name__)

# Global variable to store the data
TRAJECTORIES_DATA: Dict[str, List[Dict[str, Any]]] = {}


def load_trajectories(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load all trajectory files from a directory."""
    trajectories = {}
    for file_path in Path(directory).glob("*.jsonl"):
        # Parse repo name and revision from filename
        match = re.match(r"(.+)@([^@]+)\.jsonl", file_path.name)
        if not match:
            continue

        repo_name, revision = match.groups()

        # Load the trajectory data
        messages = []
        with open(file_path, "r") as f:
            for line in f:
                try:
                    messages.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        trajectories[file_path.name] = messages

    return trajectories


def get_github_repo_url(repo_name: str, commit_sha: str) -> str:
    """Generate GitHub URL for the repository at a specific revision."""
    return f"https://github.com/{repo_name}/tree/{commit_sha}"


def analyze_trajectory(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze a trajectory to extract key information."""
    has_commands = False
    failed_commands = []
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    # Check the last message for commands history
    if messages and messages[-1]["node"] == "commands_history":
        has_commands = True
        # Extract failed commands
        for cmd in messages[-1].get("commands", []):
            if not cmd or cmd["exit_code"] != 0:
                failed_commands.append(cmd)

    # Calculate total cost and tokens
    for message in messages:
        cost_info = calculate_message_cost(message)
        total_cost += cost_info["cost"]
        total_input_tokens += cost_info["input_tokens"]
        total_output_tokens += cost_info["output_tokens"]

    return {
        "message_count": len(messages),
        "has_commands": has_commands,
        "failed_commands": failed_commands,
        "total_cost": round(total_cost, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }


# Home page template
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trajectory Viewer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .traj-row {
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .traj-row:hover {
            background-color: #f8f9fa;
        }
        .index-badge {
            font-family: monospace;
            background-color: #6c757d;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .failed-commands {
            color: #dc3545;
            font-size: 0.9em;
        }
        .cost-badge {
            background-color: #198754;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .token-badge {
            background-color: #0d6efd;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .summary-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container py-4">
        {% set total_cost = namespace(value=0.0) %}
        {% set total_tokens = namespace(value=0) %}
        {% for filename, messages in trajectories.items() %}
            {% set analysis = analyze_trajectory(messages) %}
            {% set total_cost.value = total_cost.value + analysis.total_cost %}
            {% set total_tokens.value = total_tokens.value + analysis.total_tokens %}
        {% endfor %}
        
        <div class="summary-card">
            <h5>Summary</h5>
            <div class="row">
                <div class="col-md-4">
                    <strong>Total Trajectories:</strong> {{ trajectories|length }}
                </div>
                <div class="col-md-4">
                    <strong>Total Cost:</strong> ${{ "%.4f"|format(total_cost.value) }}
                </div>
                <div class="col-md-4">
                    <strong>Total Tokens:</strong> {{ "{:,}".format(total_tokens.value) }}
                </div>
            </div>
            {% if trajectories|length > 0 %}
            <div class="row mt-2">
                <div class="col-md-4">
                </div>
                <div class="col-md-4">
                    <strong>Average Cost:</strong> ${{ "%.4f"|format(total_cost.value / trajectories|length) }}
                </div>
                <div class="col-md-4">
                    <strong>Average Tokens:</strong> {{ "{:,}".format(total_tokens.value / trajectories|length) }}
                </div>
            </div>
            {% endif %}
        </div>

        <h4 class="mb-4">Trajectories</h4>
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Index</th>
                        <th>Repository</th>
                        <th>Revision</th>
                        <th>Messages</th>
                        <th>Tokens</th>
                        <th>Cost</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for filename, messages in trajectories.items() %}
                    {% set repo_name, revision = filename.replace('.jsonl', '').split('@') %}
                    {% set analysis = analyze_trajectory(messages) %}
                    <tr class="traj-row" onclick="window.location='/view/{{ loop.index0 }}'">
                        <td><span class="index-badge">{{ loop.index0 }}</span></td>
                        <td>{{ repo_name }}</td>
                        <td><code>{{ revision[:8] }}</code></td>
                        <td>{{ analysis.message_count }}</td>
                        <td>
                            <span class="token-badge" title="Input: {{ analysis.total_input_tokens }}, Output: {{ analysis.total_output_tokens }}">
                                {{ "{:,}".format(analysis.total_tokens) }}
                            </span>
                        </td>
                        <td>
                            <span class="cost-badge">${{ "%.4f"|format(analysis.total_cost) }}</span>
                        </td>
                        <td>
                            {% if analysis.has_commands %}
                                {% if analysis.failed_commands %}
                                    <span class="badge bg-danger">{{ analysis.failed_commands|length }} Failed Commands</span>
                                {% else %}
                                    <span class="badge bg-success">Completed</span>
                                {% endif %}
                            {% else %}
                                <span class="badge bg-warning">No Commands</span>
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

# Trajectory view template
TRAJ_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trajectory View</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .navbar {
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            padding: 0.5rem 1rem;
        }
        .message-container {
            margin: 20px 0;
        }
        .message {
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .message-agent {
            background-color: #e3f2fd;
            border-left: 4px solid #1976d2;
        }
        .message-tools {
            background-color: #f5f5f5;
            border-left: 4px solid #616161;
        }
        .message-commands {
            background-color: #fff3e0;
            border-left: 4px solid #ef6c00;
        }
        .command-item {
            padding: 8px;
            border-radius: 4px;
            margin: 5px 0;
            font-family: monospace;
        }
        .command-success {
            background-color: #e8f5e9;
            border: 1px solid #66bb6a;
        }
        .command-failed {
            background-color: #ffebee;
            border: 1px solid #ef5350;
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
        .tool-call {
            background-color: #f3e5f5;
            border-radius: 4px;
            padding: 10px;
            margin: 5px 0;
        }
        .code-block {
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .timestamp {
            color: #757575;
            font-size: 0.85em;
        }
        .cost-badge {
            background-color: #198754;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .token-badge {
            background-color: #0d6efd;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .model-badge {
            background-color: #6f42c1;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .message-metadata {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
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
                    <span class="text-primary">{{ repo_name }}</span>
                </h4>
                <p class="mb-0">
                    Revision: <code>{{ revision }}</code>
                    <a href="{{ get_github_repo_url(repo_name, revision) }}" 
                       class="btn btn-sm btn-outline-secondary ms-2" target="_blank">
                        View on GitHub
                    </a>
                </p>
            </div>
            <div class="btn-group">
                {% if prev_idx is not none %}
                    <a href="/view/{{ prev_idx }}" class="btn btn-outline-primary">Previous</a>
                {% endif %}
                {% if next_idx is not none %}
                    <a href="/view/{{ next_idx }}" class="btn btn-outline-primary">Next</a>
                {% endif %}
            </div>
        </div>
        
        <div class="message-container">
            {% for message in messages %}
                {% if message.node == "agent" %}
                    {% for msg in message.messages %}
                        <div class="message message-agent">
                            <div class="message-metadata">
                                <span class="badge bg-primary">AGENT</span>
                                <span class="timestamp">{{ message.timestamp }}</span>
                                {% if msg.message_content.get("usage_metadata") %}
                                    {% set metadata = msg.message_content.usage_metadata %}
                                    <span class="token-badge" title="Input: {{ metadata.input_tokens }}, Output: {{ metadata.output_tokens }}">
                                        {{ metadata.input_tokens + metadata.output_tokens }} tokens
                                    </span>
                                    {% if msg.response_metadata and msg.response_metadata.model_name %}
                                        <span class="model-badge">{{ msg.response_metadata.model_name }}</span>
                                        {% set cost_info = calculate_message_cost(message) %}
                                        <span class="cost-badge">${{ "%.4f"|format(cost_info.cost) }}</span>
                                    {% endif %}
                                {% endif %}
                            </div>
                            {% if msg.message_content.get("content") %}
                                <div class="message-text">{{ msg.message_content.content }}</div>
                            {% endif %}
                            {% if msg.message_content.get("tool_calls") %}
                                {% for tool_call in msg.message_content.tool_calls %}
                                    <div class="tool-call">
                                        <strong>{{ tool_call.name }}</strong>
                                        <div class="code-block">{{ tool_call.args | tojson(indent=2) }}</div>
                                    </div>
                                {% endfor %}
                            {% endif %}
                        </div>
                    {% endfor %}
                {% elif message.node == "tools" %}
                    <div class="message message-tools">
                        <div class="d-flex justify-content-between mb-2">
                            <span class="badge bg-secondary">TOOLS</span>
                            <span class="timestamp">{{ message.timestamp }}</span>
                        </div>
                        <div class="code-block">{{ message.messages[0].message_content.content }}</div>
                    </div>
                {% elif message.node == "commands_history" %}
                    <div class="message message-commands">
                        <div class="d-flex justify-content-between mb-2">
                            <span class="badge bg-warning">COMMANDS</span>
                            <span class="timestamp">{{ message.timestamp }}</span>
                        </div>
                        {% for cmd in message.commands %}
                            <div class="command-item {{ 'command-success' if cmd.exit_code == 0 else 'command-failed' }}">
                                <div class="d-flex justify-content-between">
                                    <code>{{ cmd.command }}</code>
                                    <span class="badge {{ 'bg-success' if cmd.exit_code == 0 else 'bg-danger' }}">
                                        Exit: {{ cmd.exit_code }}
                                    </span>
                                </div>
                            </div>
                        {% endfor %}
                    </div>
                {% endif %}
            {% endfor %}
        </div>
    </div>

    <script>
        // Navigation keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            {% if prev_idx is not none %}
                if (event.key === 'ArrowLeft') {
                    window.location.href = '/view/{{ prev_idx }}';
                }
            {% endif %}
            {% if next_idx is not none %}
                if (event.key === 'ArrowRight') {
                    window.location.href = '/view/{{ next_idx }}';
                }
            {% endif %}
            if (event.key === 'h') {
                window.location.href = '/';
            }
        });
    </script>
</body>
</html>
"""


def generate_trajectories_html(trajectories_data: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate HTML string for the trajectories data home page.

    Args:
        trajectories_data: Dictionary mapping filenames to lists of trajectory messages

    Returns:
        str: HTML string for the home page
    """
    with app.app_context():
        return render_template_string(
            HOME_TEMPLATE,
            trajectories=trajectories_data,
            analyze_trajectory=analyze_trajectory,
        )


def generate_trajectories_html_from_hf(
    traj_dir: str,
    repo_id: str,
    no_cache: bool = False,
) -> str:
    """Generate HTML string for trajectories data loaded from Hugging Face.

    Args:
        traj_dir: Path to the trajectories directory in the HF repo
        repo_id: Hugging Face repository ID
        no_cache: Whether to bypass cache and force redownload

    Returns:
        str: HTML string for the home page
    """
    # Get the directory path, downloading if necessary
    dir_path = get_dir_path(
        dir_path=traj_dir,
        repo_id=repo_id,
        no_cache=no_cache,
    )

    # Load data
    trajectories_data = load_trajectories(dir_path)

    return generate_trajectories_html(trajectories_data)


@app.route("/")
def index():
    return render_template_string(
        HOME_TEMPLATE,
        trajectories=TRAJECTORIES_DATA,
        analyze_trajectory=analyze_trajectory,
    )


@app.route("/view/<int:idx>")
def view_trajectory(idx: int):
    filenames = list(TRAJECTORIES_DATA.keys())
    if idx < 0 or idx >= len(filenames):
        return redirect(url_for("index"))

    filename = filenames[idx]
    messages = TRAJECTORIES_DATA[filename]
    repo_name, revision = filename.replace(".jsonl", "").split("@")

    prev_idx = idx - 1 if idx > 0 else None
    next_idx = idx + 1 if idx < len(filenames) - 1 else None

    return render_template_string(
        TRAJ_TEMPLATE,
        messages=messages,
        repo_name=repo_name,
        revision=revision,
        index=idx,
        prev_idx=prev_idx,
        next_idx=next_idx,
        get_github_repo_url=get_github_repo_url,
        calculate_message_cost=calculate_message_cost,
    )


def main():
    parser = argparse.ArgumentParser(description="View trajectory data in a web interface")
    parser.add_argument("traj_dir", type=str, help="Path to the trajectories directory")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--repo", type=str, help="Hugging Face repository to download from")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache and force redownload")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    args = parser.parse_args()

    # Get the directory path, downloading if necessary
    dir_path = get_dir_path(
        dir_path=args.traj_dir,
        repo_id=args.repo if args.repo else None,
        no_cache=args.no_cache,
    )

    # Load data
    global TRAJECTORIES_DATA
    TRAJECTORIES_DATA = load_trajectories(dir_path)

    url = f"http://{args.host}:{args.port}"
    print(f"Loaded {len(TRAJECTORIES_DATA)} trajectories. Starting server at {url}")

    # Only open browser in the main process, not in the reloader
    if not args.no_browser and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        webbrowser.open(url)

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
