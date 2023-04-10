import torch
from unittest.mock import patch
from utils.algorithms.policy_loss import clipped_surrogate_objective


@patch("utils.algorithms.policy_loss.exp")
@patch("utils.algorithms.policy_loss.sub")
def test_mocked_ratio_calculation(exp_mock, sub_mock):
    log_current_probs = torch.zeros(3)
    log_old_probs = torch.zeros(3)
    advantages = torch.zeros(3)
    epsilon = 0.2

    exp_mock.return_value = torch.ones(3)
    sub_mock.return_value = torch.zeros(3)

    clipped_surrogate_objective(log_current_probs, log_old_probs, advantages, epsilon)

    exp_mock.assert_called()
    sub_mock.assert_called()


def test_clipped_surrogate_objective():
    log_current_probs = torch.tensor([0.0, 0.1, 0.2])
    log_old_probs = torch.tensor([-0.1, 0.0, 0.1])
    advantages = torch.tensor([1.0, 2.0, 3.0])
    epsilon = 0.2

    result = clipped_surrogate_objective(
        log_current_probs, log_old_probs, advantages, epsilon
    )

    assert result.dim() == 0, "Result should be a scalar tensor"
    assert result.item() > 0, "Result should be greater than 0"
