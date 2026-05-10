import yaml

class ConfigLoader:
    def __init__(self, path: str):
        self.path = path
        self.config = self._load()

    def _load(self):
        with open(self.path, "r") as f:
            return yaml.safe_load(f)

    def get(self, *keys, default=None):
        cfg = self.config

        for key in keys:
            if not isinstance(cfg, dict) or key not in cfg:
                return default
            cfg = cfg[key]

        return cfg
