import unittest.mock as mock
import torch
from gymnasium import Env
from utils.algorithms.wrappers import ModelWrapper
from utils.algorithms.rollouts import compute_trajectories


def test_compute_trajectories_zero_steps():
    env_mock = mock.MagicMock(spec=Env)
    policy_wrapper_mock = mock.MagicMock(spec=ModelWrapper)
    value_wrapper_mock = mock.MagicMock(spec=ModelWrapper)
    reward_func_mock = mock.MagicMock()

    (
        states,
        actions,
        log_probs,
        rewards,
        values,
        num_steps_per_trajectory,
    ) = compute_trajectories(
        env=env_mock,
        state_parser=lambda x: x,
        policy_wrapper=policy_wrapper_mock,
        value_wrapper=value_wrapper_mock,
        reward_func=reward_func_mock,
        num_trajectories=0,
        num_steps=0,
    )

    assert len(states) == 0
    assert len(actions) == 0
    assert len(log_probs) == 0
    assert len(rewards) == 0
    assert len(values) == 0
    assert len(num_steps_per_trajectory) == 0


def test_compute_trajectories_single_step():
    env_mock = mock.MagicMock(spec=Env)
    env_mock.reset.return_value = ([0], False)
    env_mock.step.return_value = ([1], 1, False, False, {})

    policy_wrapper_mock = mock.MagicMock()
    policy_wrapper_mock.network = mock.MagicMock(
        return_value=(torch.tensor(0), torch.tensor(0))
    )
    policy_wrapper_mock.output_parser.side_effect = lambda x: x

    value_wrapper_mock = mock.MagicMock()
    value_wrapper_mock.network = mock.MagicMock(return_value=torch.tensor(0))
    value_wrapper_mock.output_parser.side_effect = lambda x: x

    reward_func_mock = mock.MagicMock()
    reward_func_mock.side_effect = lambda _, x: x

    (
        states,
        actions,
        log_probs,
        rewards,
        values,
        num_steps_per_trajectory,
    ) = compute_trajectories(
        env=env_mock,
        state_parser=lambda x: torch.tensor(x),
        policy_wrapper=policy_wrapper_mock,
        value_wrapper=value_wrapper_mock,
        reward_func=reward_func_mock,
        num_trajectories=1,
        num_steps=1,
    )

    assert len(states) == 1
    assert len(actions) == 1
    assert len(log_probs) == 1
    assert len(rewards) == 1
    assert len(values) == 1
    assert len(num_steps_per_trajectory) == 1


def test_compute_trajectories_multiple():
    env_mock = mock.MagicMock(spec=Env)
    env_mock.reset.return_value = ([0], False)
    env_mock.step.return_value = ([1], 1, False, False, {})

    policy_wrapper_mock = mock.MagicMock()
    policy_wrapper_mock.network = mock.MagicMock(
        return_value=(torch.tensor(0), torch.tensor(0))
    )
    policy_wrapper_mock.output_parser.side_effect = lambda x: x

    value_wrapper_mock = mock.MagicMock()
    value_wrapper_mock.network = mock.MagicMock(return_value=torch.tensor(0))
    value_wrapper_mock.output_parser.side_effect = lambda x: x

    reward_func_mock = mock.MagicMock()
    reward_func_mock.side_effect = lambda _, x: x

    (
        states,
        actions,
        log_probs,
        rewards,
        values,
        num_steps_per_trajectory,
    ) = compute_trajectories(
        env=env_mock,
        state_parser=lambda x: torch.tensor(x),
        policy_wrapper=policy_wrapper_mock,
        value_wrapper=value_wrapper_mock,
        reward_func=reward_func_mock,
        num_trajectories=3,
        num_steps=3,
    )

    assert len(states) == 3
    assert len(actions) == 3
    assert len(log_probs) == 3
    assert len(rewards) == 3
    assert len(values) == 3
    assert len(num_steps_per_trajectory) == 3
    assert num_steps_per_trajectory[0] == 3
