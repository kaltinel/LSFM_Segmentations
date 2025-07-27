import yaml
import pathlib

_CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


class Config:

    config = None

    def __init__(self):
        self.config = self.read_yaml(str(_CONFIG_PATH))

    def read_yaml(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
