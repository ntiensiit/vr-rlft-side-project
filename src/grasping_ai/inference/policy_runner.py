from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

PolicyActionSampler = Callable[[np.ndarray], np.ndarray]


def load_rl_policy_checkpoint(
    checkpoint_path: Path, device: str
) -> dict[str, torch.Tensor]:
    """Load an RL policy checkpoint from disk.

    Args:
        checkpoint_path: Path to the RL policy checkpoint file.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A mapping from parameter names to tensors representing the policy.
    """
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return cast(dict[str, torch.Tensor], checkpoint)


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
    from grasping_ai.models.rl_policy import build_policy_network, read_rl_policy_metadata

    model_state = cast(dict[str, torch.Tensor], checkpoint.get("model_state_dict"))
    metadata = read_rl_policy_metadata(cast(dict[str, object], checkpoint))
    if metadata is not None:
        ckpt_obs_dim, ckpt_action_dim, hidden_dim, num_layers = metadata
        if ckpt_obs_dim != observation_dim:
            raise ValueError(
                f"checkpoint observation_dim ({ckpt_obs_dim}) does not match "
                f"requested observation_dim ({observation_dim})"
            )
        if ckpt_action_dim != action_dim:
            raise ValueError(
                f"checkpoint action_dim ({ckpt_action_dim}) does not match "
                f"requested action_dim ({action_dim})"
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
    if isinstance(policy, torch.nn.Module):
        if model_state is not None:
            policy.load_state_dict(
                cast(dict[str, Any], model_state)
            )
        policy.to(torch.device(device))
        policy.eval()

    def runner(observation: np.ndarray) -> np.ndarray:
        """Map a single observation to an action."""
        if not isinstance(observation, np.ndarray):
            raise TypeError("observation must be a numpy array")
        if observation.ndim != 1 or observation.shape[0] != observation_dim:
            raise ValueError(
                f"observation must have shape ({observation_dim},), "
                f"got {observation.shape}"
            )

        obs_tensor = (
            torch.from_numpy(observation)
            .float()
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            action_tensor = policy(obs_tensor)
        return action_tensor.squeeze(0).cpu().numpy()

    return runner


def run_policy_step(
    runner: PolicyActionSampler, observation: np.ndarray
) -> np.ndarray:
    """Run a single inference step of an RL policy.

    Args:
        runner: Callable returned by ``build_rl_policy_runner``.
        observation: Current observation as a numpy array.

    Returns:
        The action selected by the policy as a numpy array.
    """
    return runner(observation)
