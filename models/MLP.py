from torch import nn
from typing import Tuple, List


class MLP(nn.Module):
    '''A simple Multi-Layer Perceptron (MLP) model.

    Attributes:
        stack (nn.Sequential): A stack of layers that make up the MLP model.
    '''
    stack: nn.Sequential

    def __init__(self, architecture: List[Tuple[int, str, dict]]):
        assert len(architecture) > 1, 'Architecture must have at least 2 layers.'
        super(MLP, self).__init__()
        layers = []
        input_shape, activation_name, activation_params = architecture[0]
        for output_shape, activation_name, activation_params in architecture[1:]:
            layers += [nn.Linear(input_shape, output_shape)]
            if activation_name:
                activation = getattr(nn, activation_name)
                layers += [activation(**activation_params)]
            input_shape = output_shape
        self.stack = nn.Sequential(*layers)

    def forward(self, state):
        return self.stack(state)
