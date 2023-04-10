from torch import Tensor, exp, sub, clamp, mul, min, mean


def clipped_surrogate_objective(
    log_current_probs: Tensor, log_old_probs: Tensor, advantages: Tensor, epsilon: float
) -> Tensor:
    """Compute the clipped surrogate objective for a batch of experiences.

    Args:
        log_current_probs (Tensor): Tensor of shape [num_actions]
                                containing the current policy probabilities
                                for each action in each episode.

        log_old_probs (Tensor): Tensor of shape [num_actions]
                                containing the previous policy probabilities for
                                each action in each episode.

        advantages (Tensor): Tensor of shape [sequence_length]
                             containing the computed advantages for each step
                             in each episode.

        epsilon (float: Clipping parameter for the ratio of new to
                        old probabilities

    Returns:
        Tensor: Scalar tensor representing the computed clipped surrogate
                objective.
    """
    # Calculate the probability ratios
    ratios = exp(sub(log_current_probs, log_old_probs))

    # Calculate the clipped ratios using torch.clamp
    clipped_ratios = clamp(ratios, 1 - epsilon, 1 + epsilon)
    # Compute the surrogate objectives for unclipped and clipped ratios
    surrogate_unclipped = mul(ratios, advantages)
    surrogate_clipped = mul(clipped_ratios, advantages)
    # Calculate the minimum of the unclipped and clipped surrogate objectives
    surrogate_min = min(surrogate_unclipped, surrogate_clipped)

    # Calculate the final loss by taking the negative mean of the minimum
    # surrogate objectives
    return mean(surrogate_min)
