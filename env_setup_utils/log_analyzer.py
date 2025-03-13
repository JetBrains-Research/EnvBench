import argparse
import os
import random

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage  # Use SystemMessage to set up assistant context
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()


def load_files(file_paths):
    """Reads the content of each file in file_paths."""
    contents = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            contents.append(file.read())
    return contents


def summarize_log(log_content, llm):
    """Summarizes and identifies issues in a single log file using ChatOpenAI."""
    # Define a system message for context
    system_prompt = (
        "You are an expert assistant for analyzing and summarizing agent log files. "
        "Your goal is to identify the main unexpected struggles encountered by the agent that it has failed to overcome and highlight up to 5 primary issues, "
        "such as inability to set java version, environment variables not working and other unexpected errors."
    )
    human_message = "Please summarize the following log file:\n\n" + log_content
    response = llm([SystemMessage(content=system_prompt), HumanMessage(content=human_message)])
    return response.content


def summarize_multiple_summaries(summaries, llm):
    """Further summarizes multiple summaries into a single final summary."""
    combined_summary = "\n".join(summaries)
    system_prompt = (
        "You are an expert assistant tasked with creating a final summary from multiple log summaries. "
        "Please provide an overall overview of key issues, highlighting common struggles and major points needing attention."
    )
    human_message = "Here are the combined summaries:\n\n" + combined_summary
    response = llm([SystemMessage(content=system_prompt), HumanMessage(content=human_message)])
    return response.content


def main():
    parser = argparse.ArgumentParser(description="Summarize and analyze agent logs.")
    parser.add_argument("path", type=str, help="Path to file or directory of logs.")
    parser.add_argument("-n", type=int, default=1, help="Number of random files to process if a directory is given.")
    args = parser.parse_args()

    # Initialize the ChatOpenAI model from LangChain (configure with your API key)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Handle single file or directory input
    if os.path.isfile(args.path):
        file_paths = [args.path]
    elif os.path.isdir(args.path):
        all_files = [
            os.path.join(args.path, f) for f in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, f))
        ]
        file_paths = random.sample(all_files, min(args.n, len(all_files)))
    else:
        print("Invalid path provided. Please provide a valid file or directory path.")
        return

    # Load log content from files and process each with LLM
    contents = load_files(file_paths)
    summaries = [summarize_log(content, llm) for content in contents]

    # Generate final summary if multiple files
    if len(summaries) > 1:
        final_summary = summarize_multiple_summaries(summaries, llm)
        print("Final Summary:")
        print(final_summary)
    else:
        print("Summary of the log:")
        print(summaries[0])


if __name__ == "__main__":
    main()
