from typing import Tuple
from torch import Tensor, zeros_like


def generalized_advantage_estimation(
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
