from models.MLP import MLP


def test_mlp():
    # Instantiate the MLP model
    model = MLP(
        architecture=[
            (4, "", {}),
            (64, "ReLU", {}),
            (2, "LogSoftmax", {"dim": -1}),
        ]
    )
    state_dict = model.state_dict()
    print(state_dict)
    assert len(state_dict) == 4
