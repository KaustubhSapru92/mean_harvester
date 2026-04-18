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
            cfg = cfg.get(key, {})
        return cfg if cfg else default
