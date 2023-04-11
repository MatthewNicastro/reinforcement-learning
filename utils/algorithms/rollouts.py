from gymnasium import Env
from torch import stack, tensor
from utils.algorithms.wrappers import ModelWrapper
from typing import Tuple, Callable


def compute_trajectories(
    env: Env,
    state_parser: Callable,
    policy_wrapper: ModelWrapper,
    value_wrapper: ModelWrapper,
    reward_func: Callable,
    num_trajectories: int,
    num_steps: int,
) -> Tuple[list, list, list, list, list, list]:
    """
    Compute trajectories for the given environment, policy, and value networks.

    Args:
        env (Env): Gym environment.
        state_parser (Callable): Function to preprocess state inputs.
        policy_wrapper (ModelWrapper): Wrapper for the policy network.
        value_wrapper (ModelWrapper): Wrapper for the value network.
        reward_func (Callable): Function to compute rewards.
        num_trajectories (int): Number of trajectories to compute.
        num_steps (int): Max number of steps to compute.

    Returns:
        tuple: Tuple containing:
            - states (list): List of state tensors for each trajectory.
            - actions (list): List of action tensors for each trajectory.
            - log_probs (list): List of log probability tensors for each trajectory.
            - rewards (list): List of reward tensors for each trajectory.
            - values (list): List of value tensors for each trajectory.
            - num_steps_per_trajectory (list): List of the number of steps for each trajectory.
    """
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    num_steps_per_trajectory = []
    for _ in range(num_trajectories):
        state = env.reset()[0]
        done = False
        (
            curr_rewards,
            curr_states,
            curr_actions,
            curr_probs,
            curr_values,
        ) = _init_trajectory()
        # Compute trajectories
        for _ in range(num_steps):
            state_t = state_parser(state)
            action_log_prob = policy_wrapper.network(state=state_t)
            action, log_prob = policy_wrapper.output_parser(action_log_prob)
            action, log_prob = action.item(), log_prob.item()
            new_state, reward, done, truncated, info = env.step(action)
            value_output = value_wrapper.network(state=state_t)
            value = value_wrapper.output_parser(value_output)
            value = value.item()
            curr_rewards.append(reward_func(curr_rewards, reward))
            curr_states.append(state_t)
            curr_actions.append(action)
            curr_probs.append(log_prob)
            curr_values.append(value)
            state = new_state

            if done:
                break
        num_steps_per_trajectory.append(len(curr_states))
        states.append(stack(curr_states))
        actions.append(tensor(curr_actions))
        log_probs.append(tensor(curr_probs))
        rewards.append(tensor(curr_rewards))
        values.append(tensor(curr_values))

    return (
        states,
        actions,
        log_probs,
        rewards,
        values,
        num_steps_per_trajectory,
    )


def _init_trajectory() -> Tuple[list, list, list, list, list]:
    """
    Initialize a trajectory with empty lists for rewards, states, actions, probabilities, and values.

    Returns:
        tuple: Tuple of empty lists (rewards, states, actions, probabilities, values).
    """
    return ([], [], [], [], [])
