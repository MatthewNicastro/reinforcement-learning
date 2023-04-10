import torch
import pytest
from unittest.mock import MagicMock, patch
from utils.algorithms.advantage_estimation import (
    generalized_advantage_estimation_per_trajectory,
    generalized_advantage_estimation,
)


def test_zero_rewards_values():
    rewards = torch.zeros(10)
    values = torch.zeros(10)
    gamma = 0.99
    lam = 0.95

    advantages, returns = generalized_advantage_estimation_per_trajectory(
        rewards=rewards, values=values, gamma=gamma, lam=lam
    )

    assert torch.allclose(advantages, torch.zeros_like(advantages))
    assert torch.allclose(returns, torch.zeros_like(returns))


def test_constant_rewards_zero_values():
    rewards = torch.ones(10)
    values = torch.zeros(10)
    gamma = 0.99
    lam = 0.95

    advantages, returns = generalized_advantage_estimation_per_trajectory(
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

    generalized_advantage_estimation_per_trajectory(
        rewards=mock_rewards_tensor, values=mock_values_tensor, gamma=gamma, lam=lam
    )

    mock_rewards_tensor.__getitem__.assert_called()
    mock_values_tensor.__getitem__.assert_called()


@pytest.mark.parametrize("gamma, lam", [(0.9, 0.9), (0.99, 0.95), (0.99, 0.999)])
def test_varying_gamma_lambda(gamma, lam):
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    values = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])

    advantages, returns = generalized_advantage_estimation_per_trajectory(
        rewards=rewards, values=values, gamma=gamma, lam=lam
    )

    assert torch.all(advantages > 0)
    assert torch.all(returns > 0)


@patch(
    "utils.algorithms.advantage_estimation.generalized_advantage_estimation_per_trajectory"
)
def test_generalized_advantage_estimation_empty_inputs(mock_gae):
    mock_gae.return_value = (torch.tensor([0.0]), torch.tensor([0.0]))

    advantages, returns = generalized_advantage_estimation(
        num_trajectories=0,
        rewards=[],
        values=[],
        discount_factor=0.99,
        gae_lambda=0.95,
    )

    assert len(advantages) == 0
    assert len(returns) == 0


@patch(
    "utils.algorithms.advantage_estimation.generalized_advantage_estimation_per_trajectory"
)
def test_generalized_advantage_estimation_single_trajectory(mock_gae):
    rewards = [torch.tensor([1.0, 1.0, 1.0])]
    values = [torch.tensor([0.5, 0.5, 0.5])]
    expected_advantages = [torch.tensor([1.0, 1.0, 1.0])]
    expected_returns = [torch.tensor([3.0, 2.0, 1.0])]

    mock_gae.side_effect = [
        (expected_advantages[0], expected_returns[0]),
    ]

    advantages, returns = generalized_advantage_estimation(
        num_trajectories=1,
        rewards=rewards,
        values=values,
        discount_factor=0.99,
        gae_lambda=0.95,
    )

    assert len(advantages) == 1
    assert len(returns) == 1
    assert (advantages[0] == expected_advantages[0]).all()
    assert (returns[0] == expected_returns[0]).all()


@patch(
    "utils.algorithms.advantage_estimation.generalized_advantage_estimation_per_trajectory"
)
def test_generalized_advantage_estimation_multiple_trajectories(mock_gae):
    rewards = [
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([2.0, 2.0, 2.0]),
    ]
    values = [
        torch.tensor([0.5, 0.5, 0.5]),
        torch.tensor([1.0, 1.0, 1.0]),
    ]
    expected_advantages = [
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([2.0, 2.0, 2.0]),
    ]
    expected_returns = [
        torch.tensor([3.0, 2.0, 1.0]),
        torch.tensor([6.0, 4.0, 2.0]),
    ]

    mock_gae.side_effect = [
        (expected_advantages[0], expected_returns[0]),
        (expected_advantages[1], expected_returns[1]),
    ]

    advantages, returns = generalized_advantage_estimation(
        num_trajectories=2,
        rewards=rewards,
        values=values,
        discount_factor=0.99,
        gae_lambda=0.95,
    )

    assert len(advantages) == 2
    assert len(returns) == 2
    assert (advantages[0] == expected_advantages[0]).all()
    assert (returns[0] == expected_returns[0]).all()
    assert (advantages[1] == expected_advantages[1]).all()
    assert (returns[1] == expected_returns[1]).all()
