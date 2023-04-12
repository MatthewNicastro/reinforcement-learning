from torch import nn
from typing import Callable
from dataclasses import dataclass, asdict


class ModelWrapper:
    """
    Wrapper class for PyTorch models and output parsers.

    Attributes:
        network (nn.Module): PyTorch model.
        output_parser (Callable): Function to parse model output.
    """

    def __init__(
        self,
        network: nn.Module,
        output_parser: Callable,
    ):
        """
        Initializes a ModelWrapper instance.

        Args:
            network (nn.Module): PyTorch model.
            output_parser (Callable): Function to parse model output, (outputs, index=None) -> (action, probs)
        """
        self.network = network
        self.output_parser = output_parser


@dataclass
class ConfigWrapper:
    """
    Dataclass to wrap PyTorch model configuration and output parser.

    Attributes:
        network_config (dict): PyTorch model configuration.
        output_parser (callable): Function to parse model output.
    """

    network_config: dict
    output_parser: callable

    def to_dict(self) -> dict:
        """
        Converts ConfigWrapper instance to a dictionary.

        Returns:
            dict: Dictionary representing ConfigWrapper instance.
        """
        return asdict(self)

    def to_model(self, network: nn.Module) -> ModelWrapper:
        """
        Converts ConfigWrapper instance to ModelWrapper instance.

        Args:
            network (nn.Module): PyTorch model.

        Returns:
            ModelWrapper: Instance of ModelWrapper with network and output_parser.
        """
        return ModelWrapper(network, self.output_parser)
