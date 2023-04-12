from torch import nn, optim
from typing import Callable
from wrappers.base import ModelWrapper


class ModelTrainingWrapper(ModelWrapper):
    """
    A wrapper class that adds functionality for training a PyTorch model.

    Attributes:
        network (nn.Module): The PyTorch model to be trained.
        optimizer (optim.Optimizer): The optimizer used for training the model.
        output_parser (Callable): A function that processes the model outputs.

    Methods:
        __init__(self, network: nn.Module, optimizer_name: str, optimizer_params: dict,
                 output_parser: Callable)
            Initializes a new instance of the ModelTrainingWrapper class.

        update(self, loss, retain_graph=False)
            Updates the model parameters using backpropagation.
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer_name: str,
        optimizer_params: dict,
        output_parser: Callable,
    ):
        """
        Initializes a wrapper for a PyTorch model that includes training-specific functionality such as an optimizer.

        Args:
            network (nn.Module): The PyTorch model to wrap.
            optimizer_name (str): The name of the PyTorch optimizer class to use.
            optimizer_params (dict): The parameters to pass to the optimizer class constructor.
            output_parser (Callable): A function that takes the model's outputs and an optional index, and returns a
                                      tuple of parsed outputs (outputs, index=None) -> (action, probs).

        Returns:
            None.
        """
        super().__init__(network, output_parser)
        optimizer = getattr(optim, optimizer_name)
        self.optimizer = optimizer(self.network.parameters(), **optimizer_params)

    def update(self, loss, retain_graph=False):
        """
        Updates the model parameters using backpropagation.

        Args:
            loss: The loss value to backpropagate.
            retain_graph (bool): Whether to retain the computation graph.
        """

        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
