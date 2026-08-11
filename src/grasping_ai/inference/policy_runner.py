from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

PolicyActionSampler = Callable[[np.ndarray], np.ndarray]


def load_rl_policy_checkpoint(checkpoint_path: Path, device: str) -> dict[str, torch.Tensor]:
    """Load an RL policy checkpoint from disk.

    Args:
        checkpoint_path: Path to the RL policy checkpoint file.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A mapping from parameter names to tensors representing the policy.
    """
    raise NotImplementedError


def build_rl_policy_runner(
    checkpoint: dict[str, torch.Tensor],
    observation_dim: int,
    action_dim: int,
    device: str,
) -> PolicyActionSampler:
    """Build a callable that maps observations to robot actions via an RL policy.

    Args:
        checkpoint: Loaded policy parameters.
        observation_dim: Dimensionality of the observation vector.
        action_dim: Dimensionality of the action vector.
        device: Device identifier on which inference runs.

    Returns:
        A function that takes an observation as a numpy array and returns an
        action as a numpy array.
    """
    raise NotImplementedError


def run_policy_step(runner: PolicyActionSampler, observation: np.ndarray) -> np.ndarray:
    """Run a single inference step of an RL policy.

    Args:
        runner: Callable returned by ``build_rl_policy_runner``.
        observation: Current observation as a numpy array.

    Returns:
        The action selected by the policy as a numpy array.
    """
    raise NotImplementedError
