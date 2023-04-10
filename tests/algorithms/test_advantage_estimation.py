import torch
import pytest
from unittest.mock import MagicMock
from utils.algorithms.advantage_estimation import generalized_advantage_estimation


def test_zero_rewards_values():
    rewards = torch.zeros(10)
    values = torch.zeros(10)
    gamma = 0.99
    lam = 0.95

    advantages, returns = generalized_advantage_estimation(
        rewards=rewards, values=values, gamma=gamma, lam=lam
    )

    assert torch.allclose(advantages, torch.zeros_like(advantages))
    assert torch.allclose(returns, torch.zeros_like(returns))


def test_constant_rewards_zero_values():
    rewards = torch.ones(10)
    values = torch.zeros(10)
    gamma = 0.99
    lam = 0.95

    advantages, returns = generalized_advantage_estimation(
        rewards=rewards, values=values, gamma=gamma, lam=lam
    )

    assert torch.all(advantages > 0)
    assert torch.all(returns > 0)


def test_rewards_values_with_mock():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    values = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])

    mock_rewards_tensor = MagicMock(spec=torch.Tensor)
    mock_rewards_tensor.shape = rewards.shape
    mock_rewards_tensor.__getitem__.side_effect = lambda idx: rewards[idx]

    mock_values_tensor = MagicMock(spec=torch.Tensor)
    mock_values_tensor.shape = values.shape
    mock_values_tensor.__getitem__.side_effect = lambda idx: values[idx]

    gamma = 0.99
    lam = 0.95

    generalized_advantage_estimation(
        rewards=mock_rewards_tensor, values=mock_values_tensor, gamma=gamma, lam=lam
    )

    mock_rewards_tensor.__getitem__.assert_called()
    mock_values_tensor.__getitem__.assert_called()


@pytest.mark.parametrize("gamma, lam", [(0.9, 0.9), (0.99, 0.95), (0.99, 0.999)])
def test_varying_gamma_lambda(gamma, lam):
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    values = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])

    advantages, returns = generalized_advantage_estimation(
        rewards=rewards, values=values, gamma=gamma, lam=lam
    )

    assert torch.all(advantages > 0)
    assert torch.all(returns > 0)
