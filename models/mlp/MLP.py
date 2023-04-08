from typing import TypeVar, Type
from torch import nn

T = TypeVar("T")


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        num_layers: int = 1,
        activation: Type[T] = nn.ReLU,
    ):
        super.__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(input_shape, input_shape), activation()]
        layers += [nn.Linear(input_shape, output_shape)]
        self.stack = nn.Sequential(*layers)

    def forward(self, state):
        return self.stack(state)
