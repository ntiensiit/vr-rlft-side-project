"""Reinforcement-learning policy networks."""

from __future__ import annotations

from grasping_ai.models.mlp import build_tanh_mlp

from grasping_ai.config.flattened_yaml_config import FLATTENED_YAML_CONFIG

from grasping_ai.utils.path_validation import require_path

from typing import TYPE_CHECKING, Any

import torch

POINT_CLOUD_NDIM = int(FLATTENED_YAML_CONFIG.get("geometry.point_cloud_ndim", 2))

if TYPE_CHECKING:
    from pathlib import Path

PolicyNetwork = torch.nn.Sequential
ValueNetwork = torch.nn.Sequential

RL_CHECKPOINT_FORMAT_VERSION = 1
RL_POLICY_ARCHITECTURE = "mlp"

def save_rl_policy_checkpoint(
    policy: torch.nn.Module,
    policy_checkpoint_path: Path,
    epoch: int,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    num_layers: int,
    seed: int | None = None,
) -> None:
    """Persist an RL policy under the standard checkpoint schema.

    The checkpoint carries explicit architecture metadata
    (``observation_dim``, ``action_dim``, ``hidden_dim``, ``num_layers``,
    ``architecture``, ``format_version``) so that inference can load it
    without guessing hyperparameters from parameter names.

    Args:
        policy: Policy module whose parameters should be saved.
        policy_checkpoint_path: Destination file path for the checkpoint.
        epoch: Number of completed update steps recorded in the checkpoint.
        observation_dim: Dimensionality of the policy observation vector.
        action_dim: Dimensionality of the policy action vector.
        hidden_dim: Hidden width of the policy network.
        num_layers: Number of hidden layers in the policy network.
        seed: Optional training seed recorded in the checkpoint.

    Raises:
        ValueError: If any dimension is non-positive.
        TypeError: If ``policy_checkpoint_path`` is not a ``pathlib.Path``
            instance.
    """
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    require_path(policy_checkpoint_path, "policy_checkpoint_path")

    checkpoint: dict[str, Any] = {
        "format_version": RL_CHECKPOINT_FORMAT_VERSION,
        "architecture": RL_POLICY_ARCHITECTURE,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "epoch": epoch,
        "model_state_dict": policy.state_dict(),
    }
    if seed is not None:
        checkpoint["seed"] = seed

    policy_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, policy_checkpoint_path)

def read_rl_policy_metadata(
    checkpoint: dict[str, object],
) -> tuple[int, int, int, int] | None:
    """Read standard policy metadata from a checkpoint.

    Args:
        checkpoint: Loaded RL policy checkpoint dictionary.

    Returns:
        A tuple ``(observation_dim, action_dim, hidden_dim, num_layers)`` when
        the checkpoint carries the standard metadata, otherwise ``None``.
    """
    if not isinstance(checkpoint, dict):
        return None
    keys = ("observation_dim", "action_dim", "hidden_dim", "num_layers")
    if not all(key in checkpoint for key in keys):
        return None
    try:
        dims = []
        for key in keys:
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                value = value.item()
            dims.append(int(value))  # type: ignore[call-overload]
        return tuple(dims)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None

def _sequential_linear_layers(module: torch.nn.Module) -> list[torch.nn.Linear]:
    """Return ``Linear`` submodules in ``Sequential`` child order.

    Args:
        module: Sequential module whose direct children are inspected.

    Returns:
        ``Linear`` layers in module child order.

    Raises:
        TypeError: If ``module`` is not a ``torch.nn.Sequential`` instance.
    """
    if not isinstance(module, torch.nn.Sequential):
        raise TypeError("module must be a torch.nn.Sequential instance")
    return [child for child in module if isinstance(child, torch.nn.Linear)]

def build_sb3_net_arch(hidden_dim: int, num_layers: int) -> dict[str, list[int]]:
    """Build SB3 ``net_arch`` matching ``build_policy_network`` depth.

    Args:
        hidden_dim: Width of each hidden layer in the policy and value nets.
        num_layers: Number of hidden layers to allocate in each network.

    Returns:
        SB3 ``policy_kwargs['net_arch']`` dictionary with matching ``pi`` and
        ``vf`` hidden-layer lists.

    Raises:
        ValueError: If ``hidden_dim`` or ``num_layers`` is non-positive.
    """
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    hidden_layers = [hidden_dim] * num_layers
    return {"pi": hidden_layers, "vf": hidden_layers}

def copy_sb3_policy_weights(
    sb3_policy: torch.nn.Module,
    legacy_policy: torch.nn.Module,
) -> None:
    """Copy Stable-Baselines3 MlpPolicy weights into a legacy ``Sequential`` policy.

    SB3 stores hidden layers in ``mlp_extractor.policy_net`` and the action
    head in ``action_net``. The legacy policy from ``build_policy_network``
    interleaves ``Linear`` and ``Tanh`` layers; this helper maps all hidden
    ``Linear`` layers dynamically instead of relying on fixed indices.

    Args:
        sb3_policy: Trained SB3 policy module (``model.policy``).
        legacy_policy: Target policy returned by ``build_policy_network``.

    Raises:
        TypeError: If either argument is not a ``torch.nn.Module``.
        ValueError: If expected SB3 submodules are missing or layer counts differ.
    """
    if not isinstance(sb3_policy, torch.nn.Module):
        raise TypeError("sb3_policy must be a torch.nn.Module instance")
    if not isinstance(legacy_policy, torch.nn.Module):
        raise TypeError("legacy_policy must be a torch.nn.Module instance")

    policy_net = getattr(getattr(sb3_policy, "mlp_extractor", None), "policy_net", None)
    action_net = getattr(sb3_policy, "action_net", None)
    if policy_net is None or action_net is None:
        raise ValueError("sb3_policy must expose mlp_extractor.policy_net and action_net")
    if not isinstance(action_net, torch.nn.Linear):
        raise TypeError("SB3 action_net must be a Linear layer")

    sb3_hidden = _sequential_linear_layers(policy_net)
    legacy_linears = _sequential_linear_layers(legacy_policy)
    if len(legacy_linears) != len(sb3_hidden) + 1:
        raise ValueError(
            f"Legacy policy has {len(legacy_linears)} Linear layers but SB3 policy_net "
            f"has {len(sb3_hidden)} hidden layers",
        )

    legacy_hidden = legacy_linears[:-1]
    legacy_output = legacy_linears[-1]
    for sb3_layer, legacy_layer in zip(sb3_hidden, legacy_hidden, strict=True):
        if sb3_layer.weight.shape != legacy_layer.weight.shape:
            raise ValueError(
                f"Shape mismatch copying SB3 layer {sb3_layer.weight.shape} "
                f"to legacy layer {legacy_layer.weight.shape}",
            )
        legacy_layer.weight.data.copy_(sb3_layer.weight.data)
        legacy_layer.bias.data.copy_(sb3_layer.bias.data)

    if action_net.weight.shape != legacy_output.weight.shape:
        raise ValueError(
            f"Shape mismatch copying SB3 action_net {action_net.weight.shape} "
            f"to legacy output {legacy_output.weight.shape}",
        )
    legacy_output.weight.data.copy_(action_net.weight.data)
    legacy_output.bias.data.copy_(action_net.bias.data)

def build_policy_network(observation_dim: int, action_dim: int, hidden_dim: int, num_layers: int) -> PolicyNetwork:
    """Construct a policy network mapping observations to action distributions.

    Args:
        observation_dim: Dimensionality of the policy observation vector.
        action_dim: Dimensionality of the action vector.
        hidden_dim: Width of the hidden layers.
        num_layers: Number of hidden layers in the policy network.

    Returns:
        A callable policy mapping a batched observation tensor of shape
        ``(B, observation_dim)`` to action parameters of shape
        ``(B, action_dim)``.
    """
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    return build_tanh_mlp(observation_dim, hidden_dim, action_dim, num_layers)

def build_value_network(observation_dim: int, hidden_dim: int, num_layers: int) -> ValueNetwork:
    """Construct a value network for actor-critic style algorithms.

    Args:
        observation_dim: Dimensionality of the observation vector.
        hidden_dim: Width of the hidden layers.
        num_layers: Number of hidden layers in the value network.

    Returns:
        A callable value network mapping observations to scalar values.
    """
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    return build_tanh_mlp(observation_dim, hidden_dim, 1, num_layers)

def select_action(
    policy: PolicyNetwork,
    observation: torch.Tensor,
    rng: torch.Generator,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """Sample an action from a stochastic policy given an observation.

    Args:
        policy: Policy network returned by ``build_policy_network``.
        observation: Observation tensor with shape ``(B, observation_dim)``.
        rng: Torch random generator used to sample actions.
        noise_scale: Standard deviation of the Gaussian exploration noise
            added to the deterministic policy output.

    Returns:
        A sampled action tensor with shape ``(B, action_dim)``.
    """
    if observation.ndim != POINT_CLOUD_NDIM:
        raise ValueError(f"observation must have shape (B, obs_dim), got {observation.shape}")
    if not isinstance(rng, torch.Generator):
        raise TypeError("rng must be a torch.Generator instance")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")

    action_mean = policy(observation)
    noise = torch.randn(
        action_mean.shape,
        generator=rng,
        device=action_mean.device,
        dtype=action_mean.dtype,
    )
    return action_mean + noise_scale * noise
