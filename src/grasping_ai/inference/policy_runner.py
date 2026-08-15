from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grasping_ai.training.checkpoint_io import (
    load_torch_checkpoint,
    read_checkpoint_model_state_dict,
)

PolicyActionSampler = Callable[[np.ndarray], np.ndarray]


def load_rl_policy_checkpoint(checkpoint_path: Path, device: str) -> dict[str, Any]:
    """Load an RL policy checkpoint from disk.

    Args:
        checkpoint_path: Path to the RL policy checkpoint file.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        Deserialized checkpoint dictionary including ``model_state_dict``.

    Raises:
        TypeError: If ``checkpoint_path`` is not a ``pathlib.Path`` instance.
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If ``torch.load`` fails or the payload is not a dictionary.
    """
    return load_torch_checkpoint(checkpoint_path, device)


def build_rl_policy_runner(
    checkpoint: dict[str, Any],
    observation_dim: int,
    action_dim: int,
    device: str,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    stochastic: bool = False,
    exploration_noise: float = 0.1,
    seed: int | None = None,
) -> PolicyActionSampler:
    """Build a callable that maps observations to robot actions via an RL policy.

    Args:
        checkpoint: Loaded policy checkpoint dictionary.
        observation_dim: Dimensionality of the observation vector.
        action_dim: Dimensionality of the action vector.
        device: Device identifier on which inference runs.
        action_low: Optional per-dimension lower bound used to clip returned
            actions. ``None`` disables clipping.
        action_high: Optional per-dimension upper bound used to clip returned
            actions. ``None`` disables clipping.
        stochastic: When ``True``, sample actions with Gaussian exploration
            noise via ``select_action`` instead of deterministic forward pass.
        exploration_noise: Standard deviation of exploration noise when
            ``stochastic`` is enabled.
        seed: Optional random seed for stochastic action sampling.

    Returns:
        A function that takes an observation as a numpy array and returns an
        action as a numpy array clipped to ``[action_low, action_high]`` when
        both bounds are supplied.
    """
    from grasping_ai.models.rl_policy import (
        build_policy_network,
        read_rl_policy_metadata,
        select_action,
    )

    model_state = read_checkpoint_model_state_dict(checkpoint)
    metadata = read_rl_policy_metadata(checkpoint)
    if metadata is not None:
        ckpt_obs_dim, ckpt_action_dim, hidden_dim, num_layers = metadata
        if ckpt_obs_dim != observation_dim:
            raise ValueError(
                f"checkpoint observation_dim ({ckpt_obs_dim}) does not match "
                f"requested observation_dim ({observation_dim})"
            )
        if ckpt_action_dim != action_dim:
            raise ValueError(
                f"checkpoint action_dim ({ckpt_action_dim}) does not match requested action_dim ({action_dim})"
            )
    else:
        # Legacy checkpoints carry no metadata; infer from parameter names.
        hidden_dim = 64
        num_layers = 2
        if model_state is not None:
            if "0.weight" in model_state:
                hidden_dim = model_state["0.weight"].shape[0]
            weight_keys = [k for k in model_state if k.endswith(".weight")]
            num_layers = max(1, len(weight_keys) - 1)

    policy = build_policy_network(observation_dim, action_dim, hidden_dim, num_layers)
    if model_state is not None:
        policy.load_state_dict(model_state)
    policy.to(torch.device(device))
    policy.eval()

    clip_low = None if action_low is None else np.asarray(action_low, dtype=np.float64)
    clip_high = None if action_high is None else np.asarray(action_high, dtype=np.float64)
    if clip_low is not None and clip_low.shape != (action_dim,):
        raise ValueError(f"action_low must have shape ({action_dim},), got {clip_low.shape}")
    if clip_high is not None and clip_high.shape != (action_dim,):
        raise ValueError(f"action_high must have shape ({action_dim},), got {clip_high.shape}")

    device_obj = torch.device(device)
    action_rng = None
    if stochastic:
        action_rng = torch.Generator(device=device_obj)
        action_rng.manual_seed(seed if seed is not None else 0)

    def runner(observation: np.ndarray) -> np.ndarray:
        """Map a single observation to an action clipped to the actuator bounds."""
        if not isinstance(observation, np.ndarray):
            raise TypeError("observation must be a numpy array")
        if observation.ndim != 1 or observation.shape[0] != observation_dim:
            raise ValueError(f"observation must have shape ({observation_dim},), got {observation.shape}")

        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device_obj)
        with torch.no_grad():
            if stochastic and action_rng is not None:
                action_tensor = select_action(policy, obs_tensor, action_rng, noise_scale=exploration_noise)
            else:
                action_tensor = policy(obs_tensor)
        action = action_tensor.squeeze(0).cpu().numpy()
        if clip_low is not None and clip_high is not None:
            return np.clip(action, clip_low, clip_high)
        return action

    return runner


def run_policy_step(runner: PolicyActionSampler, observation: np.ndarray) -> np.ndarray:
    """Run a single inference step of an RL policy.

    Args:
        runner: Callable returned by ``build_rl_policy_runner``.
        observation: Current observation as a numpy array.

    Returns:
        The action selected by the policy as a numpy array.
    """
    return runner(observation)
