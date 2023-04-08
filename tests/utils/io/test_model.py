import pytest
from unittest.mock import patch
from pathlib import Path
from utils.io.model import save_model, load_model
import torch.nn as nn


class MockModel(nn.Module):
    def __init__(self, input: int):
        super(MockModel, self).__init__()
        self.input = input
        self.layer = nn.Sequential(nn.Linear(input, 1), nn.ReLU())


@pytest.fixture
def mock_input():
    return 5


@pytest.fixture
def mock_model(mock_input):
    return MockModel(input=mock_input)


@pytest.fixture
def mock_path():
    return Path("/save")


@patch("utils.io.model.save")
def test_save_model(mocked_save, mock_model, mock_path):
    save_model(mock_model, mock_path)
    mocked_save.assert_called()


@pytest.mark.parametrize("eval_mode", [True, False])
@patch("utils.io.model.load")
def test_load(mocked_load, eval_mode, mock_model, mock_path, mock_input):
    expected_state_dict = mock_model.state_dict()
    mocked_load.return_value = expected_state_dict
    model = load_model(MockModel, mock_path, eval_mode=eval_mode, input=mock_input)
    state_dict = model.state_dict()
    mocked_load.assert_called()
    assert len(expected_state_dict) == len(state_dict)
    for param in state_dict:
        assert param in expected_state_dict
        assert state_dict[param].size() == expected_state_dict[param].size()
    assert model.training == (not eval_mode)
