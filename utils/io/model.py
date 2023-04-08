from typing import TypeVar, Type
from torch import save, load
from torch.nn import Module
from pathlib import Path

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
    state_dict_path: Path = None,
    eval_mode: bool = True,
    **kwargs: dict
) -> T:
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
    model: Module = model_class(**kwargs)
    state_dict = load(state_dict_path)
    model.load_state_dict(state_dict)
    if eval_mode:
        model.eval()
    return model
