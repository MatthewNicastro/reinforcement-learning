from pathlib import Path
from dill import load, dump
from wrappers.base import ConfigWrapper


def dump_config(config: dict, path: Path):
    """
    Serialize a Python dictionary into a binary file using pickle.

    Args:
        config (dict): The Python dictionary to serialize.
        path (Path): The path to the output binary file.
    """
    with open(path, "wb") as f:
        dump(config, f)


def load_config(path: Path) -> ConfigWrapper:
    """
    Deserialize a Python dictionary from a binary file using pickle.

    Args:
        path (Path): The path to the input binary file.

    Returns:
        config (dict): The deserialized Python dictionary.
    """

    with open(path, "rb") as f:
        config = load(f)
    return ConfigWrapper(**config)
