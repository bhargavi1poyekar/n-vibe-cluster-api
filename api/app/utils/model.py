import pickle
from pathlib import Path
import os
import yaml


def find_model(name: str) -> Path:
    cwd = os.getcwd()
    config_folder = "api/app/config"
    config_file = os.path.join(cwd, config_folder, f"{name}.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_model(name: str):
    cwd = os.getcwd()
    model_config = find_model(name)
    model_path = os.path.join(
        cwd, "api/app/models/weights", model_config["name"], model_config["model"]
    )
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        print(model)
    return model


if __name__ == "__main__":
    load_model("biped_hq")
