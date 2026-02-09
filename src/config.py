import argparse
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class ModelSettings:
    name: str
    temperature: float = 0.8
    # top_k: Optional[int] = None
    # top_p: Optional[int] = None


@dataclass
class Config:
    language: str
    main_model: ModelSettings
    rerank_model: ModelSettings
    answer_model: ModelSettings
    gcp_project: str
    max_retries: int = 5
    log_level: str = "INFO"
    embed_metadata: bool = False
    embedding_api_base: str = ""
    embedding_model: str = ""
    rerank_metadata: bool = True


parser = argparse.ArgumentParser(
    description="SQL agent to answer question and perform post-processing on DB data"
)
parser.add_argument(
    "--config", type=str, default="config.yml", help="Path to the config file"
)
# Add command line arguments that can override config values
parser.add_argument("--language", type=str, help="Language for the agent")
parser.add_argument("--max_retries", type=int, help="Maximum number of retries")
parser.add_argument("--main_model_name", type=str, help="Model name to use")
parser.add_argument("--gcp_project", type=str, help="GCP project ID")
parser.add_argument("--provider", type=str, help="LLM inference API provider name")
parser.add_argument("--log_level", type=str, help="Logging level")
parser.add_argument(
    "--embed_metadata",
    type=bool,
    help="Wether to compute embeddings of tables' metadata or not",
)
parser.add_argument(
    "--embedding_api_base",
    type=str,
    help="API base url for the text embedding endpoint",
)
parser.add_argument(
    "--embedding_model",
    type=str,
    help="Model identifier for the selected embedding model",
)
parser.add_argument("--temperature", type=float, help="Temperature for model sampling")
parser.add_argument(
    "--question", type=str, help="Question to ask the agent (single query mode)"
)
parser.add_argument(
    "--rerank_metadata",
    type=bool,
    help="Wether to call an LLM to prune metadata if it's too long or not",
)
parser.add_argument(
    "--rerank_model_name",
    type=str,
    help="Model identifier for the reranker model pruning metadata",
)
parser.add_argument(
    "--answer_model_name",
    type=str,
    help="Model identifier for the model generating the final answer",
)

args = parser.parse_args()


def get_args_parser():
    global args
    return args


def _parse_config_file(filepath: str = "config.yml") -> Config:
    """Reads the YAML config file into a dictionary and returns it"""

    def _get_model_settings(model_settings_key: str) -> ModelSettings | None:
        try:
            settings = config.pop(model_settings_key)
            return ModelSettings(**settings)
        except KeyError:
            return None

    with open(filepath, "r") as file:
        config = yaml.safe_load(file)

    # Check for missing required fields
    required_fields = [
        "language",
        "main_model",
        "gcp_project",
    ]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")

    main_model_settings: ModelSettings = _get_model_settings(
        "main_model"
    )  # type: ignore
    rerank_settings = _get_model_settings("rerank_model") or main_model_settings
    answer_settings = _get_model_settings("answer_model") or main_model_settings

    return Config(
        **config,
        main_model=main_model_settings,
        rerank_model=rerank_settings,
        answer_model=answer_settings,
    )


def _override_config_with_args(config: Config, args: argparse.Namespace) -> Config:
    """Override config values with command line arguments if provided."""
    # Map command line argument names to config field names
    arg_to_config = {
        "language": "language",
        "max_retries": "max_retries",
        "main_model_name": "main_model.name",
        "temperature": "main_model.temperature",
        "gcp_project": "gcp_project",
        "provider": "provider",
        "log_level": "log_level",
        "embed_metadata": "embed_metadata",
        "embedding_api_base": "embedding_api_base",
        "embedding_model": "embedding_model",
        "rerank_metadata": "rerank_metadata",
        "rerank_model_name": "rerank_model.name",
        "answer_model_name": "answer_model.name",
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


def read_config():
    config = _parse_config_file(args.config)
    return _override_config_with_args(config, args)
