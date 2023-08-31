# config.py
import json

class Config:
    def __init__(self, config_path="src/configs/default.json"):
        with open(config_path, 'r') as f:
            self._config = json.load(f)

        self.dataset = self._config["dataset"]
        self.visualization = self._config["visualization"]
        self.dataloader = self._config["dataloader"]
        self.training = self._config["training"]
        self.network = self._config["network"]

    def get(self, key, default=None):
        return self._config.get(key, default)

