from torch import Tensor, zeros_like


def generalized_advantage_estimation(
    rewards: Tensor, values: Tensor, next_values: Tensor, gamma=0.99, tau=0.95
) -> Tensor:
    """
        Compute the generalized advantage estimation (GAE) for a batch of
        experiences.

    Args:
        rewards (Tensor): Tensor of shape [batch_size, sequence_length]
                          containing the rewards for each step in each episode.

        values (Tensor): Tensor of shape [batch_size, sequence_length]
                         containing the estimated values for each step in each
                         episode.

        next_values (Tensor): Tensor of shape [batch_size, sequence_length]
                              containing the estimated values of the next
                              states for each step in each episode.

        gamma (float, optional): Discount factor for future rewards. Defaults
                                 to 0.99.

        tau (float, optional): Parameter controlling the trade-off between
                               bias and variance in GAE. Defaults to 0.95.

    Returns:
        Tensor: Tensor of shape [batch_size, sequence_length] containing the
                computed advantages for each step in each episode.
    """
    td_errors = rewards + gamma * next_values - values
    gae = 0
    advantages = zeros_like(td_errors)
    last_idx = td_errors.shape[-1] - 1
    for idx in range(last_idx, -1, -1):
        gae = td_errors[idx] + gamma * tau * gae
        advantages[idx] = gae
    return advantages
