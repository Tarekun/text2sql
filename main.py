from langchain.messages import HumanMessage
from src.agent.graph import Text2SqlAgent
from src.config import read_config, Config
from src.db import gcp_pull_metadata, get_table_metadata
from src.logger import configure_logger, logger
from src.utils import print_graph
import argparse


def override_config_with_args(config: Config, args: argparse.Namespace) -> Config:
    """Override config values with command line arguments if provided."""
    # Map command line argument names to config field names
    arg_to_config = {
        "language": "language",
        "max_retries": "max_retries",
        "model_name": "model_name",
        "gcp_project": "gcp_project",
        "provider": "provider",
        "log_level": "log_level",
        "temperature": "model_settings.temperature",
    }

    config_copy = config.__dict__.copy()
    for arg_name, config_path in arg_to_config.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            # Handle nested attributes (like model_settings.temperature)
            if "." in config_path:
                parts = config_path.split(".")
                current = config_copy
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = getattr(args, arg_name)
            else:
                config_copy[config_path] = getattr(args, arg_name)

    # Reconstruct the config object
    return Config(**config_copy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to the config file"
    )
    # Add command line arguments that can override config values
    parser.add_argument("--language", type=str, help="Language for the agent")
    parser.add_argument("--max_retries", type=int, help="Maximum number of retries")
    parser.add_argument("--model_name", type=str, help="Model name to use")
    parser.add_argument("--gcp_project", type=str, help="GCP project ID")
    parser.add_argument("--provider", type=str, help="LLM inference API provider name")
    parser.add_argument("--log_level", type=str, help="Logging level")
    parser.add_argument(
        "--temperature", type=float, help="Temperature for model sampling"
    )

    args = parser.parse_args()

    config = read_config(args.config)
    config = override_config_with_args(config, args)

    configure_logger(config)
    # has to be after the configure_logger call
    logger.debug(f"Loaded config: {config}")
    agent = Text2SqlAgent(config)
    print_graph(agent.graph)

    while True:
        question = input("> ")
        if question == "/quit":
            break

        answer = agent.invoke(question)
        print(answer)
    print("Bye!")
