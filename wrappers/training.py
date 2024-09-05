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
    optimizer: optim.Optimizer

    def __init__(
        self,
        network: nn.Module,
        optimizer_name: str,
        optimizer_params: dict,
        output_parser: Callable,
    ):
        super().__init__(network, output_parser)
        optimizer = getattr(optim, optimizer_name)
        self.optimizer = optimizer(
            self.network.parameters(),
            **optimizer_params,
        )

    def update(self, loss, retain_graph: bool = False):
        """
        Updates the model parameters using backpropagation.

        Args:
            loss: The loss value to backpropagate.
            retain_graph (bool): Whether to retain the computation graph.
        """

        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
