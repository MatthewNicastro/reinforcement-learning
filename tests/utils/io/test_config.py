from unittest.mock import mock_open, patch
from pathlib import Path
from dill import dumps
from wrappers.base import ConfigWrapper
from utils.io.config import dump_config, load_config


def test_dump_config():
    test_config = {"key": "value"}
    test_path = Path("test_config.pkl")
    m = mock_open()

    with patch("builtins.open", m):
        dump_config(test_config, test_path)

    m.assert_called_once_with(test_path, "wb")


def test_load_config():
    test_config = {"network_config": "value", "output_parser": lambda x: x}
    test_path = Path("test_config.pkl")
    m = mock_open(read_data=dumps(test_config))

    with patch("builtins.open", m):
        loaded_config = load_config(test_path)

    assert isinstance(loaded_config, ConfigWrapper)
    assert loaded_config.network_config == test_config["network_config"]
    m.assert_called_once_with(test_path, "rb")
