from typing import Tuple
from torch import Tensor, zeros_like


def generalized_advantage_estimation_per_trajectory(
    rewards: Tensor, values: Tensor, gamma: float, lam: float
) -> Tuple[Tensor, Tensor]:
    """
        Compute the generalized advantage estimation (GAE) for a batch of
        experiences.

    Args:
        rewards (Tensor): Tensor of shape [sequence_length]
                          containing the rewards for each step.

        values (Tensor): Tensor of shape [sequence_length]
                         containing the estimated values for each step.

        gamma (float): Discount factor for future rewards.

        lam (float): GAE lambda parameter for weighting the importance of
                     future advantages.

    Returns:
        (Tensor, Tensor): Tensor of shape [sequence_length] containing the
                          computed advantages, and return.
    """
    returns = zeros_like(rewards)
    advantages = zeros_like(rewards)
    next_return = 0
    next_value = 0
    for idx in range(returns.shape[-1] - 1, -1, -1):
        reward = rewards[idx]
        value = values[idx]
        delta = reward + (gamma * next_value) - value
        advantages[idx] = delta + gamma * lam * next_value
        next_return = reward + gamma * next_return
        returns[idx] = next_return
    return advantages, returns


def generalized_advantage_estimation(
    num_trajectories: int,
    rewards: list,
    values: list,
    discount_factor: float,
    gae_lambda: float,
) -> Tuple[list, list]:
    """
    Compute generalized advantage estimation (GAE) for a batch of trajectories.

    Args:
        num_trajectories (int): Number of trajectories in the batch.
        rewards (list): List of reward tensors for each trajectory.
        values (list): List of value tensors for each trajectory.
        discount_factor (float): Discount factor for future rewards.
        gae_lambda (float): GAE lambda parameter.

    Returns:
        tuple: Tuple containing:
            - advantages (list): List of advantage tensors for each trajectory.
            - returns (list): List of return tensors for each trajectory.
    """
    advantages = [None for _ in range(num_trajectories)]
    returns = [None for _ in range(num_trajectories)]

    for trajectory_num in range(num_trajectories):
        curr_rewards = rewards[trajectory_num]
        curr_values = values[trajectory_num]
        (
            trajectory_advantages,
            trajectory_returns,
        ) = generalized_advantage_estimation_per_trajectory(
            rewards=curr_rewards,
            values=curr_values,
            gamma=discount_factor,
            lam=gae_lambda,
        )
        advantages[trajectory_num] = trajectory_advantages
        returns[trajectory_num] = trajectory_returns
    return advantages, returns
