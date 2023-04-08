from typing import TypeVar, Type
from pathlib import Path
from yaml import load, FullLoader

T = TypeVar("T")


def load_config(config_class: Type[T], path: Path) -> T:
    """
    Load a configuration file in YAML format and create an instance of a
    specified class.

    Args:
        config_class (type): The class of the configuration object to create.
        path (pathlib.Path): The path to the YAML configuration file.

    Returns:
        An instance of the specified configuration class.
    """
    with open(path, "r") as f:
        yaml_data = load(f, Loader=FullLoader)
    data = config_class(**yaml_data)
    return data
