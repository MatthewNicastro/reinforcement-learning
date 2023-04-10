import pytest
from models.MLP import MLP


@pytest.mark.parametrize(
    "output_activation_name, output_activation_args",
    (("ReLU", {}), ("", {}), ("LogSoftmax", {"dim": -1})),
)
def test_mlp(output_activation_name, output_activation_args):
    input_shape = 10
    output_shape = 2
    num_layers = 3
    activation_name = "ReLU"

    # Instantiate the MLP model
    model = MLP(
        input_shape=input_shape,
        output_shape=output_shape,
        num_layers=num_layers,
        activation_name=activation_name,
        output_activation_name=output_activation_name,
        output_activation_args=output_activation_args,
    )

    state_dict = model.state_dict()
    assert len(state_dict) == 2 * (num_layers + 1)
