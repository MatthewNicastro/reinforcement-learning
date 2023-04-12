from typing import TypeVar, Type
from torch import save, load
from torch.nn import Module
from pathlib import Path
from wrappers.base import ModelWrapper
from utils.io.config import load_config

T = TypeVar("T")


def save_model(model: Module, path: Path):
    """
    Save a PyTorch model to a file.

    Args:
        model (torch.nn.Module): The model to save.
        path (pathlib.Path): The file path to save the model to.
    """
    state_dict = model.state_dict()
    save(state_dict, str(path))


def load_model(
    model_class: Type[T],
    state_dict_path: Path,
    config_path: Path,
    eval_mode: bool = True,
) -> ModelWrapper:
    """
    Load a PyTorch model from a file.

    Args:
        model_class (type): The class of the model to create.

        state_dict_path (pathlib.Path): The file path to the saved model state
                                        dictionary.

        eval_mode (bool): Whether to set the model to evaluation mode.
                          Defaults to True.

    Returns:
        T: The loaded model instance.
    """
    config_wrapper = load_config(config_path)
    model: Module = model_class(**config_wrapper.network_config)
    state_dict = load(state_dict_path)
    model.load_state_dict(state_dict)
    if eval_mode:
        model.eval()
    return config_wrapper.to_model(model)
