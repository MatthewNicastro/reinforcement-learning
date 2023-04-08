from torch import Tensor, clamp, min, mean


def clipped_surrogate_objective(
    current_probs: Tensor, old_probs: Tensor, advantages: Tensor, epsilon=0.2
) -> Tensor:
    """Compute the clipped surrogate objective for a batch of experiences.

    Args:
        current_probs (Tensor): Tensor of shape [batch_size, num_actions]
                                containing the current policy probabilities
                                for each action in each episode.

        old_probs (Tensor): Tensor of shape [batch_size, num_actions]
                            containing the previous policy probabilities for
                            each action in each episode.

        advantages (Tensor): Tensor of shape [batch_size, sequence_length]
                             containing the computed advantages for each step
                             in each episode.

        epsilon (float, optional): Clipping parameter for the ratio of new to
                                   old probabilities. Defaults to 0.2.

    Returns:
        Tensor: Scalar tensor representing the computed clipped surrogate
                objective.
    """
    # Calculate the probability ratios
    ratios = current_probs / old_probs

    # Calculate the clipped ratios using torch.clamp
    clipped_ratios = clamp(ratios, 1 - epsilon, 1 + epsilon)
    # Compute the surrogate objectives for unclipped and clipped ratios
    surrogate_unclipped = ratios * advantages
    surrogate_clipped = clipped_ratios * advantages
    # Calculate the minimum of the unclipped and clipped surrogate objectives
    surrogate_min = min(surrogate_unclipped, surrogate_clipped)

    # Calculate the final loss by taking the negative mean of the minimum
    # surrogate objectives
    return -mean(surrogate_min)
