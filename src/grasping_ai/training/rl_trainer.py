from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch

RLObservation = np.ndarray
RLAction = np.ndarray
RLReward = float
RLTransition = tuple[RLObservation, RLAction, RLReward, RLObservation, bool]
RLStepFunction = Callable[[RLAction], RLTransition]


def build_rl_training_step(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_ratio: float,
    entropy_coefficient: float,
    device: str,
) -> Callable[[list[RLTransition]], dict[str, float]]:
    """Construct a callable that performs a single RL policy update step.

    Args:
        policy: Trainable policy module.
        optimizer: Optimizer used to update the policy parameters.
        clip_ratio: PPO-style clipping ratio applied to the objective.
        entropy_coefficient: Coefficient applied to the entropy bonus.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A callable that accepts a list of collected transitions and returns
        a dictionary of training metrics for the update step.
    """
    raise NotImplementedError


def run_rl_training_loop(
    update_step: Callable[[list[RLTransition]], dict[str, float]],
    rollout_iterator: Iterator[list[RLTransition]],
    num_updates: int,
    checkpoint_path: Path,
    log_every: int,
) -> None:
    """Run a closed-loop RL training loop over rollout data.

    Args:
        update_step: Callable returned by ``build_rl_training_step``.
        rollout_iterator: Iterator yielding batches of collected transitions.
        num_updates: Number of policy update steps to perform.
        checkpoint_path: Path where the final policy checkpoint is written.
        log_every: Logging interval measured in update steps.
    """
    raise NotImplementedError


def compute_discounted_returns(transitions: list[RLTransition], gamma: float) -> np.ndarray:
    """Compute discounted returns for a list of RL transitions.

    Args:
        transitions: List of ``(obs, action, reward, next_obs, done)`` tuples.
        gamma: Discount factor in ``[0, 1]``.

    Returns:
        Discounted returns as a numpy array aligned with ``transitions``.
    """
    raise NotImplementedError


def compute_gae_advantages(
    transitions: list[RLTransition],
    value_fn: Callable[[RLObservation], float],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimation values for an RL rollout.

    Args:
        transitions: List of ``(obs, action, reward, next_obs, done)`` tuples.
        value_fn: Callable mapping observations to scalar value estimates.
        gamma: Discount factor in ``[0, 1]``.
        gae_lambda: GAE smoothing parameter in ``[0, 1]``.

    Returns:
        A tuple ``(advantages, returns)`` of numpy arrays.
    """
    raise NotImplementedError
