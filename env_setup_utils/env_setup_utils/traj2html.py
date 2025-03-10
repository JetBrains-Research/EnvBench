import argparse
import json
import webbrowser
from pathlib import Path


# Function to convert message_content type to distinct bootstrap colors
def content_type_to_color(content_type):
    colors = {
        "INIT": "primary",
        "ACTION": "info",
        "OBSERVATION": "warning",
        "TERMINATION": "primary",
        "ERROR": "danger",
    }
    return colors.get(content_type, "secondary")  # Default color is secondary


# Function to set background color based on message direction
def message_direction_to_bg(message_type):
    bg_colors = {
        "e2s": "background-color: #e0f7e0;",  # Light green
        "s2e": "background-color: #e0e7f7;",  # Light blue
    }
    return bg_colors.get(message_type, "background-color: #f8f9fa;")  # Default light gray


error_style = "background-color: #f8d7da;"  # Light red


def convert_jsonl_to_html(jsonl_path):
    # Read JSONL data from the provided file path
    with open(jsonl_path, "r", encoding="utf-8") as file:
        chat_entries = [json.loads(line) for line in file]

    # Define the HTML template with escaped curly braces in CSS
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chat Log</title>
        <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .chat-container {{
                max-width: 800px;
                margin: 20px auto;
            }}
            .message {{
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 5px;
            }}
            .message-content {{
                white-space: pre-wrap;
                background-color: #f1f1f1;
                padding: 10px;
                border-radius: 5px;
            }}
            .agent {{
                font-weight: bold;
            }}
            .code-block {{
                background-color: #f7f7f7;
                color: #333;
                padding: 10px;
                border-radius: 5px;
                overflow: auto;
            }}
            .collapsed {{
                max-height: 300px;
                overflow-y: hidden;
                position: relative;
            }}
            .collapsed::after {{
                content: "";
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 50px;
                background: linear-gradient(transparent, #f1f1f1);
                pointer-events: none;
            }}
            .toggle-button {{
                display: none;
                width: 100%;
                padding: 5px;
                margin-top: 5px;
                background-color: #e9ecef;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            }}
            .toggle-button:hover {{
                background-color: #dee2e6;
            }}
            .long-content .toggle-button {{
                display: block;
            }}
        </style>
        <script>
            function toggleContent(button) {{
                const content = button.previousElementSibling;
                const isCollapsed = content.classList.contains('collapsed');
                
                if (isCollapsed) {{
                    content.classList.remove('collapsed');
                    button.textContent = 'Show Less';
                }} else {{
                    content.classList.add('collapsed');
                    button.textContent = 'Show More';
                }}
            }}

            function initializeCollapsible() {{
                document.querySelectorAll('.message-content').forEach(content => {{
                    if (content.scrollHeight > 300) {{
                        content.classList.add('collapsed');
                        content.parentElement.classList.add('long-content');
                    }}
                }});
            }}

            window.onload = initializeCollapsible;
        </script>
    </head>
    <body>
        <div class="container chat-container">
            {chat_content}
        </div>
    </body>
    </html>
    """

    # Generate the chat content from JSONL
    chat_content = ""
    for entry in chat_entries:
        node_type = entry["node"]
        timestamp = entry["timestamp"]

        if node_type == "agent":
            for message in entry["messages"]:
                message_content = message["message_content"]

                # Handle tool calls
                if "tool_calls" in message_content:
                    for tool_call in message_content["tool_calls"]:
                        formatted_message = f"<p><strong>Tool Name:</strong> {tool_call['name']}</p>"
                        formatted_message += (
                            f"<p><strong>Tool Args:</strong> {json.dumps(tool_call['args'], indent=2)}</p>"
                        )
                        chat_content += f"""
                        <div class="message" style="{message_direction_to_bg('s2e')}">
                            <div>
                                <span class="badge badge-info">TOOL_CALL</span>
                                <span class="agent">Agent</span>
                                <span class="text-muted">{timestamp}</span>
                            </div>
                            <div class="message-content">{formatted_message}</div>
                            <button class="toggle-button" onclick="toggleContent(this)">Show More</button>
                        </div>
                        """

                # Handle regular content
                if message_content.get("content"):
                    formatted_message = f"<p>{message_content['content']}</p>"
                    chat_content += f"""
                    <div class="message" style="{message_direction_to_bg('s2e')}">
                        <div>
                            <span class="badge badge-primary">CONTENT</span>
                            <span class="agent">Agent</span>
                            <span class="text-muted">{timestamp}</span>
                        </div>
                        <div class="message-content">{formatted_message}</div>
                        <button class="toggle-button" onclick="toggleContent(this)">Show More</button>
                    </div>
                    """

        elif node_type == "tools":
            for message in entry["messages"]:
                message_content = message["message_content"]
                formatted_message = f"<div class='code-block'><pre>{message_content.get('content', '')}</pre></div>"
                chat_content += f"""
                <div class="message" style="{message_direction_to_bg('e2s')}">
                    <div>
                        <span class="badge badge-warning">TOOL_RESPONSE</span>
                        <span class="agent">Tools</span>
                        <span class="text-muted">{timestamp}</span>
                    </div>
                    <div class="message-content">{formatted_message}</div>
                    <button class="toggle-button" onclick="toggleContent(this)">Show More</button>
                </div>
                """

        elif node_type == "commands_history":
            commands = entry.get("commands", [])
            formatted_message = "<pre><code>"
            for cmd in commands:
                status = "✓" if cmd["exit_code"] == 0 else "✗"
                formatted_message += f"{status} {cmd['command']} (exit: {cmd['exit_code']})\n"
            formatted_message += "</code></pre>"
            chat_content += f"""
            <div class="message" style="{message_direction_to_bg('e2s')}">
                <div>
                    <span class="badge badge-info">COMMANDS_HISTORY</span>
                    <span class="agent">System</span>
                    <span class="text-muted">{timestamp}</span>
                </div>
                <div class="message-content">{formatted_message}</div>
                <button class="toggle-button" onclick="toggleContent(this)">Show More</button>
            </div>
            """

    # Combine the HTML template with the chat content
    html_content = html_template.format(chat_content=chat_content)

    # Write the HTML content to a file
    output_file = Path(jsonl_path).with_suffix(".html")
    output_file.write_text(html_content)

    # Open the HTML file in the default web browser
    webbrowser.open(output_file.absolute().as_uri())
    print(f"Chat log has been saved to {output_file} and opened in the browser.")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert a JSONL chat log to an HTML file with Bootstrap styling.")
    parser.add_argument("jsonl_path", type=str, help="Path to the input .jsonl file containing chat data.")
    args = parser.parse_args()

    # Convert the JSONL chat log to HTML
    convert_jsonl_to_html(args.jsonl_path)
