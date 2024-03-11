import pickle
from pathlib import Path
import os
import yaml
from typing import Any


def find_model(name: str) -> Any:
    cwd = os.getcwd()
    config_folder = "app/config"
    config_file = os.path.join(cwd, config_folder, f"{name}.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_model(name: str) -> Any:
    cwd = os.getcwd()
    model_config = find_model(name)
    model_path = os.path.join(
        cwd, "app/models/weights", model_config["name"], model_config["model"]
    )
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model
