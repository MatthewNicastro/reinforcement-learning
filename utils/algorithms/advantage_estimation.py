from typing import Tuple
from torch import Tensor, zeros


def generalized_advantage_estimation_per_trajectory(
    rewards: Tensor, values: Tensor, gamma: float, lam: float
) -> Tuple[Tensor, Tensor]:
    """
    Compute the generalized advantage estimation (GAE) of a experience.

    Args:
        rewards (Tensor): Tensor of shape [trajectory_length]
                          containing the rewards for each step.
        values (Tensor): Tensor of shape [trajectory_length]
                         containing the estimated values for each step.
        gamma (float): Discount factor for future rewards.
        lam (float): GAE lambda parameter for weighting the importance of
                     future advantages.

    Returns:
        (Tensor, Tensor): Tensor of shape [trajectory_length] containing the
                          computed advantages, and return.
    """
    # Step 1: Calculate the TD residuals (delta) for each time step in the trajectory
    trajectory_length = rewards.shape[-1]
    # Step 2: Compute the GAE advantages by weighting the importance of future advantages using the λ (lambda) parameter
    advantages = zeros((trajectory_length,))
    gae = 0
    for t in reversed(range(trajectory_length)):
        old_value_state = values[t]
        next_value_state = 0 if t + 1 >= trajectory_length else values[t + 1]
        delta = rewards[t] + gamma * next_value_state - old_value_state
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    # Step 3: Calculate the return for each time step
    returns = zeros((trajectory_length,))
    ret = values[-1]
    for t in reversed(range(trajectory_length)):
        ret = rewards[t] + gamma * ret
        returns[t] = ret

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
        (
            trajectory_advantages,
            trajectory_returns,
        ) = generalized_advantage_estimation_per_trajectory(
            rewards=rewards[trajectory_num],
            values=values[trajectory_num],
            gamma=discount_factor,
            lam=gae_lambda,
        )
        advantages[trajectory_num] = trajectory_advantages
        returns[trajectory_num] = trajectory_returns
    return advantages, returns
