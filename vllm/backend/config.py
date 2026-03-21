from pathlib import Path
import configparser

def load_config():
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_path = base_dir / "config" / "config.ini"

    config = configparser.ConfigParser()
    config.read(config_path)

    return {
        "ollama_host": config.get("ollama", "host"),
        "model_name": config.get("ollama", "model"),
        "server_host": config.get("server", "host"),
        "server_port": config.getint("server", "port"),
    }