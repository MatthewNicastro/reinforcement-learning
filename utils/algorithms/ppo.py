from tqdm import tqdm
from gymnasium import Env
from typing import Callable
from utils.algorithms.wrappers import ModelWrapper
from utils.algorithms.rollouts import compute_trajectories
from utils.algorithms.policy_loss import clipped_surrogate_objective
from utils.algorithms.advantage_estimation import generalized_advantage_estimation


def proximal_policy_optimization(
    env: Env,
    state_parser: Callable,
    policy_wrapper: ModelWrapper,
    value_wrapper: ModelWrapper,
    value_loss: Callable,
    reward_func: Callable,
    epochs: int,
    num_trajectories: int,
    num_steps: int,
    updates_per_epoch: int,
    discount_factor: float = 0.99,
    gae_lambda: float = 0.95,
    clipping_parameter: float = 0.2,
) -> dict:
    """
    Train an agent using Proximal Policy Optimization (PPO) algorithm.

    Args:
        env (Env): Gym environment.
        state_parser (Callable): Function to preprocess state inputs.
        policy_wrapper (ModelWrapper): Wrapper for the policy network.
        value_wrapper (ModelWrapper): Wrapper for the value network.
        value_loss (Callable): Function to compute value loss.
        reward_func (Callable): Function to compute rewards.
        epochs (int): Number of training epochs.
        num_trajectories (int): Number of trajectories to compute.
        num_steps (int): Max number of steps per epoch.
        updates_per_epoch (int): Number of optimization steps per epoch.
        discount_factor (float): Discount factor for future rewards. Defaults to 0.99.
        gae_lambda (float): GAE lambda parameter. Defaults to 0.95.
        clipping_parameter (float): Clipping parameter for PPO. Defaults to 0.2.

    Returns:
        logger (dict): A dictionary with the following structure
                       "policy_losses" (list): all the policy losses (epochs * updates_per_epoch),
                       "value_losses" (list): all the value losses (epochs * updates_per_epoch),,
                       "num_steps" (list): number of steps per trajectory (epochs * num_steps),
    """
    # Initialize logger
    logger = _init_logger()

    # Main training loop
    for _ in tqdm(range(epochs)):
        (
            states,
            actions,
            log_probs,
            rewards,
            values,
            num_steps_per_trajectory,
        ) = compute_trajectories(
            env,
            state_parser,
            policy_wrapper,
            value_wrapper,
            reward_func,
            num_trajectories,
            num_steps,
        )
        logger["num_steps"] += num_steps_per_trajectory

        advantages, returns = generalized_advantage_estimation(
            num_trajectories, rewards, values, discount_factor, gae_lambda
        )
        # Update policy and value functions
        for i in range(updates_per_epoch):
            policy_func_loss = 0.0
            value_func_loss = 0.0
            for trajectory_num in range(num_trajectories):
                trajectory_states_t = states[trajectory_num]
                trajectory_actions = actions[trajectory_num]
                trajectory_log_probs = log_probs[trajectory_num].detach()
                trajectory_advantages = advantages[trajectory_num]

                trajectory_advantages = (
                    trajectory_advantages - trajectory_advantages.mean()
                ) / (trajectory_advantages.std() + 1e-8)
                trajectory_new_action_log_probs = policy_wrapper.network(
                    trajectory_states_t
                )
                _, trajectory_new_log_probs = policy_wrapper.output_parser(
                    trajectory_new_action_log_probs, index=trajectory_actions
                )

                policy_func_loss -= clipped_surrogate_objective(
                    log_current_probs=trajectory_new_log_probs,
                    log_old_probs=trajectory_log_probs,
                    advantages=trajectory_advantages,
                    epsilon=clipping_parameter,
                )

                trajectory_returns = returns[trajectory_num]
                trajectory_new_value_preds = value_wrapper.network(
                    trajectory_states_t
                ).squeeze()

                value_func_loss += value_loss(
                    trajectory_new_value_preds, trajectory_returns
                )

            policy_func_loss /= num_trajectories
            value_func_loss /= num_trajectories
            policy_wrapper.update(loss=policy_func_loss)
            logger["policy_losses"].append(policy_func_loss.item())

            value_wrapper.update(loss=value_func_loss)
            logger["value_losses"].append(value_func_loss.item())
    return logger


def _init_logger():
    """
    Initialize a logger to store training statistics.

    Returns:
        dict: Dictionary containing keys and empty lists for policy_losses, value_losses, and num_steps.
    """
    return {
        "policy_losses": [],
        "value_losses": [],
        "num_steps": [],
    }
