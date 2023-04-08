from models.MLP import MLP


def test_mlp():
    input_shape = 10
    output_shape = 2
    num_layers = 3
    activation_name = "ReLU"

    # Instantiate the MLP model
    model = MLP(input_shape, output_shape, num_layers, activation_name)

    state_dict = model.state_dict()
    assert len(state_dict) == 2 * (num_layers + 1)
