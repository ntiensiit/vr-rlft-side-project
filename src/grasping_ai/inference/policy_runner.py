"""Execute RL policies during inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from grasping_ai.models.rl_policy import (
    build_policy_network,
    read_rl_policy_metadata,
    select_action,
)
from grasping_ai.training.checkpoint_io import read_checkpoint_model_state_dict

PolicyActionSampler = Callable[[np.ndarray], np.ndarray]


def _resolve_policy_architecture(
    checkpoint: dict[str, Any],
    model_state: dict[str, torch.Tensor] | None,
    observation_dim: int,
    action_dim: int,
) -> tuple[int, int]:
    """Resolve ``(hidden_dim, num_layers)`` from checkpoint metadata or weights.

    Raises:
        ValueError: If checkpoint metadata dims mismatch the requested dims.
    """
    metadata = read_rl_policy_metadata(checkpoint)
    if metadata is not None:
        ckpt_obs_dim, ckpt_action_dim, hidden_dim, num_layers = metadata
        if ckpt_obs_dim != observation_dim:
            msg = (
                f"checkpoint observation_dim ({ckpt_obs_dim}) does not match "
                f"requested observation_dim ({observation_dim})"
            )
            raise ValueError(
                msg,
            )
        if ckpt_action_dim != action_dim:
            msg = f"checkpoint action_dim ({ckpt_action_dim}) does not match requested action_dim ({action_dim})"
            raise ValueError(
                msg,
            )
        return hidden_dim, num_layers

    # Legacy checkpoints carry no metadata; infer from parameter names.
    hidden_dim = 64
    num_layers = 2
    if model_state is not None:
        if "0.weight" in model_state:
            hidden_dim = model_state["0.weight"].shape[0]
        weight_keys = [k for k in model_state if k.endswith(".weight")]
        num_layers = max(1, len(weight_keys) - 1)
    return hidden_dim, num_layers


def _resolve_clip_bounds(
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    action_dim: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate and normalize optional action clipping bounds.

    Raises:
        ValueError: If a supplied bound does not match ``(action_dim,)``.
    """
    clip_low = None if action_low is None else np.asarray(action_low, dtype=np.float64)
    clip_high = None if action_high is None else np.asarray(action_high, dtype=np.float64)
    if clip_low is not None and clip_low.shape != (action_dim,):
        msg = f"action_low must have shape ({action_dim},), got {clip_low.shape}"
        raise ValueError(msg)
    if clip_high is not None and clip_high.shape != (action_dim,):
        msg = f"action_high must have shape ({action_dim},), got {clip_high.shape}"
        raise ValueError(msg)
    return clip_low, clip_high


def _build_action_rng(*, stochastic: bool, device_obj: torch.device, seed: int | None) -> torch.Generator | None:
    """Build the exploration-noise generator for stochastic sampling."""
    if not stochastic:
        return None
    action_rng = torch.Generator(device=device_obj)
    action_rng.manual_seed(seed if seed is not None else 0)
    return action_rng


# Public API: optional inference knobs stay individual keyword arguments
# because scripts and tests pass them by name.
def build_rl_policy_runner(  # noqa: PLR0913
    checkpoint: dict[str, Any],
    observation_dim: int,
    action_dim: int,
    device: str,
    *,
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
    model_state = read_checkpoint_model_state_dict(checkpoint)
    hidden_dim, num_layers = _resolve_policy_architecture(checkpoint, model_state, observation_dim, action_dim)

    policy = build_policy_network(observation_dim, action_dim, hidden_dim, num_layers)
    if model_state is not None:
        policy.load_state_dict(model_state)
    policy.to(torch.device(device))
    policy.eval()

    clip_low, clip_high = _resolve_clip_bounds(action_low, action_high, action_dim)

    device_obj = torch.device(device)
    action_rng = _build_action_rng(stochastic=stochastic, device_obj=device_obj, seed=seed)

    def runner(observation: np.ndarray) -> np.ndarray:
        """Map a single observation to an action clipped to the actuator bounds."""
        if not isinstance(observation, np.ndarray):
            msg = "observation must be a numpy array"
            raise TypeError(msg)
        if observation.ndim != 1 or observation.shape[0] != observation_dim:
            msg = f"observation must have shape ({observation_dim},), got {observation.shape}"
            raise ValueError(msg)

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
