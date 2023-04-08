from unittest.mock import patch
from pathlib import Path
from utils.io.config import load_config
from dataclasses import dataclass


@dataclass
class MockConfig:
    name: str


@patch("utils.io.config.load")
def test_load_config(mocked_load, open_file):
    # Create a temporary YAML file with some configuration data.
    mocked_load.return_value = {"name": "Alice"}
    config_file = Path("config.yaml")
    loaded_config = load_config(MockConfig, config_file)
    assert loaded_config.name == "Alice"
