from torch import nn, optim
from typing import Callable


class ModelTrainingWrapper:
    def __init__(
        self,
        network: nn.Module,
        optimizer_name: str,
        optimizer_params: dict,
        output_parser: Callable,  # (outputs, index=None) -> Tuple
    ):
        self.network = network
        optimizer = getattr(optim, optimizer_name)
        self.optimizer = optimizer(self.network.parameters(), **optimizer_params)
        self.output_parser = output_parser

    def update(self, loss, retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
