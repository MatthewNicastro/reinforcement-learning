from torch import nn


class MLP(nn.Module):
    """A simple multi-layer perceptron (MLP) neural network implemented in PyTorch.

    This class defines a neural network with a specified number of hidden layers,
    each consisting of a linear layer and an activation function. The output layer
    is also a linear layer. The activation function used in the hidden layers is
    specified as a string (e.g. 'ReLU', 'Tanh').

    Attributes:
        stack (nn.Sequential): A PyTorch sequential model consisting of the layers
                               defined in the constructor.

    Methods:
        forward(state): Compute the forward pass of the neural network given an input
                        tensor `state`.

    Args:
        input_shape (int): The number of input dimensions to the neural network.

        output_shape (int): The number of output dimensions from the neural network.

        num_layers (int): The number of hidden layers in the neural network.

        activation_name (str): The name of the activation function to use in the hidden
                               layers of the neural network (e.g. 'ReLU', 'Tanh').
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        num_layers: int,
        activation_name: str,
    ):
        super.__init__()
        layers = []
        activation = getattr(nn, activation_name)
        for _ in range(num_layers):
            layers += [nn.Linear(input_shape, input_shape), activation()]
        layers += [nn.Linear(input_shape, output_shape)]
        self.stack = nn.Sequential(*layers)

    def forward(self, state):
        return self.stack(state)
