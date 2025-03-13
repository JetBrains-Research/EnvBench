#!/usr/bin/env python3

import argparse
import json
import os
from typing import Any, Dict, List
import webbrowser

from flask import Flask, redirect, render_template_string, url_for

from env_setup_utils.analysis.utils import get_file_path

app = Flask(__name__)

# Global variable to store the data
SCRIPTS_DATA: List[Dict[str, Any]] = []

# Home page template
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Scripts Viewer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .script-row {
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .script-row:hover {
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
    </style>
</head>
<body>
    <div class="container py-4">
        <h4 class="mb-4">Scripts Data</h4>
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Index</th>
                        <th>Repository</th>
                        <th>Revision</th>
                    </tr>
                </thead>
                <tbody>
                    {% for script in scripts %}
                    <tr class="script-row" onclick="window.location='/view/{{ loop.index0 }}'">
                        <td><span class="index-badge">{{ loop.index0 }}</span></td>
                        <td>{{ script.repository }}</td>
                        <td><code>{{ script.revision[:8] }}</code></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

# Script view template
SCRIPT_TEMPLATE = """
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
                    <span class="text-primary">{{ script.repository }}</span>
                </h4>
                <p class="mb-0">
                    Revision: <code>{{ script.revision }}</code>
                    <a href="https://github.com/{{ script.repository }}/tree/{{ script.revision }}" 
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
        
        <div class="script-container">{{ script.script }}</div>
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


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return results


def generate_scripts_html(scripts_data: List[Dict[str, Any]]) -> str:
    """Generate HTML string for the scripts data home page.

    Args:
        scripts_data: List of script entries

    Returns:
        str: HTML string for the home page
    """
    with app.app_context():
        return render_template_string(HOME_TEMPLATE, scripts=scripts_data)


def generate_scripts_html_from_hf(
    scripts_file: str,
    repo_id: str,
    no_cache: bool = False,
) -> str:
    """Generate HTML string for scripts data loaded from Hugging Face.

    Args:
        scripts_file: Path to the JSONL scripts file in the HF repo
        repo_id: Hugging Face repository ID
        no_cache: Whether to bypass cache and force redownload

    Returns:
        str: HTML string for the home page
    """
    # Get the file path, downloading if necessary
    file_path = get_file_path(
        file_path=scripts_file,
        caller_name="scripts_viewer",
        repo_id=repo_id,
        no_cache=no_cache,
    )

    # Load data
    scripts_data = load_jsonl(file_path)

    return generate_scripts_html(scripts_data)


@app.route("/")
def index():
    return render_template_string(HOME_TEMPLATE, scripts=SCRIPTS_DATA)


@app.route("/view/<int:idx>")
def view_script(idx: int):
    if idx < 0 or idx >= len(SCRIPTS_DATA):
        return redirect(url_for("index"))

    script = SCRIPTS_DATA[idx]
    prev_idx = idx - 1 if idx > 0 else None
    next_idx = idx + 1 if idx < len(SCRIPTS_DATA) - 1 else None

    return render_template_string(SCRIPT_TEMPLATE, script=script, index=idx, prev_idx=prev_idx, next_idx=next_idx)


def main():
    parser = argparse.ArgumentParser(description="View scripts data in a web interface")
    parser.add_argument("scripts_file", type=str, nargs="?", help="Path to the JSONL scripts file")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--repo", type=str, help="Hugging Face repository to download from")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache and force redownload")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    args = parser.parse_args()

    # Get the file path, downloading if necessary
    file_path = get_file_path(
        file_path=args.scripts_file,
        caller_name="scripts_viewer",
        repo_id=args.repo if args.repo else None,
        no_cache=args.no_cache,
    )

    # Load data
    global SCRIPTS_DATA
    SCRIPTS_DATA = load_jsonl(file_path)

    url = f"http://{args.host}:{args.port}"
    print(f"Loaded {len(SCRIPTS_DATA)} scripts. Starting server at {url}")

    # Only open browser in the main process, not in the reloader
    if not args.no_browser and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        webbrowser.open(url)

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
