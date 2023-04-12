import pytest
from unittest.mock import patch
from torch.nn import Module
from pathlib import Path
from wrappers.base import ModelWrapper, ConfigWrapper
from utils.io.model import save_model, load_model


class MockModel(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.eval_count = 0

    def eval(self):
        self.eval_count += 1


@pytest.fixture
def model_instance():
    return MockModel()


@patch("utils.io.model.save")
def test_save_model(mock_save, model_instance):
    test_path = Path("test_model.pth")
    save_model(model_instance, test_path)
    mock_save.assert_called_once_with(model_instance.state_dict(), str(test_path))


@pytest.mark.parametrize("eval, eval_num_calls", [(False, 0), (True, 1)])
@patch("utils.io.model.load")
@patch("utils.io.model.load_config")
def test_load_model(mock_load_config, mock_load, eval, eval_num_calls, model_instance):
    test_state_dict_path = Path("test_model.pth")
    test_config_path = Path("test_config.yml")
    mock_config = {"network_config": {"dummy_arg": 1}, "output_parser": lambda x: x}

    mock_load_config.return_value = ConfigWrapper(**mock_config)
    mock_load.return_value = model_instance.state_dict()

    loaded_model_wrapper = load_model(
        MockModel, test_state_dict_path, test_config_path, eval_mode=eval
    )

    assert isinstance(loaded_model_wrapper, ModelWrapper)

    mock_load.assert_called_once_with(test_state_dict_path)
    mock_load_config.assert_called_once_with(test_config_path)
    model_instance.eval_count = eval_num_calls
