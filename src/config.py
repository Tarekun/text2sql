from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class ModelSettings:
    temperature: float = 0.8
    # top_k: Optional[int] = None
    # top_p: Optional[int] = None


@dataclass
class Config:
    language: str
    model_name: str
    provider: str
    model_settings: ModelSettings
    max_retries: int = 5


def read_config(filepath: str = "config.yml") -> Config:
    """Reads the YAML config file into a dictionary and returns it"""
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)

    model_settings_dict = config.pop("model_settings")
    model_settings = ModelSettings(**model_settings_dict)

    return Config(**config, model_settings=model_settings)
