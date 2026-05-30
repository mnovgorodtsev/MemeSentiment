from pathlib import Path
import configparser
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def load_hyperparams(path: str) -> Dict:
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / path

    with open(config_path, "r") as f:
        hyperparameters = json.load(f)

    return hyperparameters


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.config_path = self._find_config_path()
        self.parser = configparser.ConfigParser()

        if not self.parser.read(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self._prompts_config = load_hyperparams("prompts.json")

        logger.info(f"Configuration loaded from {self.config_path}")
        self._initialized = True

    @staticmethod
    def _find_config_path() -> str:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "config" / "config.ini"

        if config_path.exists():
            return str(config_path)

        raise FileNotFoundError(f"config.ini not found in: {config_path}")

    @property
    def ollama_host(self) -> str:
        return self.parser.get("ollama", "host", fallback="http://localhost:11434")

    @property
    def ollama_model(self) -> str:
        return self.parser.get("ollama", "model", fallback="qwen3-vl:2b")

    @property
    def server_host(self) -> str:
        return self.parser.get("server", "host", fallback="0.0.0.0")

    @property
    def server_port(self) -> int:
        return self.parser.getint("server", "port", fallback=7860)

    @property
    def memotion_dataset_path(self) -> str:
        return self.parser.get("datasets", "MEMOTION_DATASET_PATH", fallback="path")
    
    @property
    def polish_dataset_path(self) -> str:
        return self.parser.get("datasets", "POLISH_DATASET_PATH", fallback="path")

    @property
    def images_base_path(self) -> str:
        return self.parser.get("datasets", "IMAGES_BASE_PATH", fallback="path")
    
    @property
    def polish_base_path(self) -> str:
        return self.parser.get("datasets", "POLISH_IMAGES_BASE_PATH", fallback="path")

    @property
    def prompts_config(self) -> Dict:
        return self._prompts_config

    @property
    def classification_config(self) -> Dict:
        return self.prompts_config.get("classification_config", {})

    @property
    def benchmark_config(self) -> Dict:
        return self.prompts_config.get("benchmark_config", {})