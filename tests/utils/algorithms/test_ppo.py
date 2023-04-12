import torch
import pytest
from unittest.mock import MagicMock
from utils.algorithms.ppo import proximal_policy_optimization


class DummyEnv:
    def reset(self):
        return [1.0], 0, False, {}

    def step(self, action):
        return [1.0], 0, False, False, {}


class DummyPolicyModelTrainingWrapper:
    def __init__(self, output):
        self.output = output

    def network(self, state):
        return self.output

    def output_parser(self, output, index=None):
        return (
            (output.to(int), output)
            if index is None
            else (output[index].to(int), output[index])
        )

    def update(self, loss, retain_graph=None):
        pass


class DummyValueModelTrainingWrapper:
    def __init__(self, output):
        self.output = output

    def network(self, state):
        return self.output

    def output_parser(self, output, index=None):
        return output if index is None else output[index]

    def update(self, loss, retain_graph=None):
        pass


def dummy_state_parser(state):
    return torch.tensor([state])


def dummy_value_loss(preds, target):
    return torch.mean(torch.square(preds - target))


def dummy_reward_func(curr_rewards, reward):
    return reward


@pytest.fixture
def ppo_args():
    env = DummyEnv()
    state_parser = dummy_state_parser
    policy_wrapper = DummyPolicyModelTrainingWrapper(torch.tensor([0.0]))
    value_wrapper = DummyValueModelTrainingWrapper(torch.tensor([0.0]))
    value_loss = dummy_value_loss
    reward_func = dummy_reward_func
    epochs = 1
    num_trajectories = 1
    num_steps = 10
    updates_per_epoch = 1

    return (
        env,
        state_parser,
        policy_wrapper,
        value_wrapper,
        value_loss,
        reward_func,
        epochs,
        num_trajectories,
        num_steps,
        updates_per_epoch,
    )


def test_proximal_policy_optimization(ppo_args):
    logger = proximal_policy_optimization(*ppo_args)

    assert "policy_losses" in logger
    assert "value_losses" in logger
    assert "num_steps" in logger
    assert len(logger["policy_losses"]) == 1
    assert len(logger["value_losses"]) == 1
    assert len(logger["num_steps"]) == 1


def test_mocked_policy_update(ppo_args):
    policy_wrapper = ppo_args[2]
    policy_wrapper.update = MagicMock()

    proximal_policy_optimization(*ppo_args)

    policy_wrapper.update.assert_called_once()


def test_mocked_value_update(ppo_args):
    value_wrapper = ppo_args[3]
    value_wrapper.update = MagicMock()

    proximal_policy_optimization(*ppo_args)

    value_wrapper.update.assert_called_once()
