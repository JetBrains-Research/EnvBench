#!/usr/bin/env python3
"""
Script to generate cadence config files from template based on input paths.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Tuple, Optional

TEMPLATE_FILE = ".cadence/configs/EnvBench multi.yaml"
CONFIG_DIR = ".cadence/configs"
GENERATED_FILES_LIST = ".cadence/configs/.generated_files"

def parse_input_path(input_str: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse input string to extract SHORT, FULL_PATH, and LONG.
    
    Expected format:
    RL-envsetup/checkpoints/RL-envsetup-baselines/baselines_Qwen3-8B_single-turn_shellcheck/2025-07-16-22-39-17-akovrigin/global_step_45/
    
    Returns:
        Tuple of (SHORT, FULL_PATH, LONG) or None if parsing fails
    """
    # Clean up the input string
    input_str = input_str.strip()
    
    # Regex pattern to extract the components
    # Matches: RL-envsetup/checkpoints/RL-envsetup-baselines/[SHORT]/[date-user]/global_step_XX/
    pattern = r'RL-envsetup/checkpoints/RL-envsetup-baselines/([^/]+)/([^/]+)/global_step_\d+/?'
    
    match = re.search(pattern, input_str)
    if not match:
        print(f"Warning: Could not parse input string: {input_str}")
        return None
    
    short = match.group(1)  # baselines_Qwen3-8B_single-turn_shellcheck
    date_user = match.group(2)  # 2025-07-16-22-39-17-akovrigin
    long = f"{short}/{date_user}"  # baselines_Qwen3-8B_single-turn_shellcheck/2025-07-16-22-39-17-akovrigin
    full_path = input_str.rstrip('/')  # Remove trailing slash if present
    
    return short, full_path, long

def generate_config_file(short: str, full_path: str, long: str) -> str:
    """
    Generate a config file based on the template.
    
    Returns:
        Path to the generated file
    """
    # Read template
    with open(TEMPLATE_FILE, 'r') as f:
        template_content = f.read()
    
    # Replace placeholders
    content = template_content.replace('SHORT', short)
    content = content.replace('FULL_PATH', full_path)
    content = content.replace('LONG', long)
    
    # Generate filename based on SHORT
    filename = f"EnvBench_{short}.yaml"
    filepath = os.path.join(CONFIG_DIR, filename)
    
    # Write the file
    with open(filepath, 'w') as f:
        f.write(content)
    
    return filepath

def track_generated_file(filepath: str):
    """Add generated file to tracking list."""
    with open(GENERATED_FILES_LIST, 'a') as f:
        f.write(filepath + '\n')

def create_configs():
    """Create config files from stdin input."""
    print("Enter paths (one per line, press Ctrl+D to finish):")
    
    generated_files = []
    
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            parsed = parse_input_path(line)
            if parsed:
                short, full_path, long = parsed
                filepath = generate_config_file(short, full_path, long)
                generated_files.append(filepath)
                track_generated_file(filepath)
                print(f"Generated: {filepath}")
            else:
                print(f"Skipped invalid input: {line}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        return
    
    print(f"\nGenerated {len(generated_files)} config files.")

def clean_configs():
    """Clean all generated config files."""
    if not os.path.exists(GENERATED_FILES_LIST):
        print("No generated files to clean.")
        return
    
    removed_count = 0
    with open(GENERATED_FILES_LIST, 'r') as f:
        for line in f:
            filepath = line.strip()
            if os.path.exists(filepath):
                os.remove(filepath)
                removed_count += 1
                print(f"Removed: {filepath}")
    
    # Remove the tracking file
    os.remove(GENERATED_FILES_LIST)
    print(f"\nCleaned {removed_count} generated files.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate cadence config files from template"
    )
    parser.add_argument(
        '--clean', 
        action='store_true',
        help="Clean all generated config files"
    )
    
    args = parser.parse_args()
    
    # Ensure config directory exists
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Check if template exists
    if not os.path.exists(TEMPLATE_FILE):
        print(f"Error: Template file {TEMPLATE_FILE} not found!")
        sys.exit(1)
    
    if args.clean:
        clean_configs()
    else:
        create_configs()

if __name__ == "__main__":
    main() 