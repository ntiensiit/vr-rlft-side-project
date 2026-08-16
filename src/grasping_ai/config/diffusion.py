"""Diffusion model configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from omegaconf import DictConfig

@dataclass(frozen=True)
class DiffusionSchedule:
    """Parameters of the linear noise schedule used by diffusion models.

    Attributes:
        beta_start: Initial noise level ``beta_0``.
        beta_end: Final noise level ``beta_{T-1}``.
        num_steps: Number of diffusion timesteps ``T``.
    """

    beta_start: float = 1e-4
    beta_end: float = 0.02
    num_steps: int = 100

    @classmethod
    def load_from_config(cls, cfg: DictConfig | None = None) -> DiffusionSchedule:
        """Load DiffusionSchedule parameters from a composed Hydra config.

        Args:
            cfg: The project configuration mapping. When ``None``, the global
                flattened YAML config is used.

        Returns:
            A DiffusionSchedule instance populated with configured parameters.
        """
        from grasping_ai.config.flattened_yaml_config import (
            FLATTENED_YAML_CONFIG,
            FlattenedYAMLConfig,
        )

        resolved_cfg = FLATTENED_YAML_CONFIG.cfg if cfg is None else cfg
        yaml_config = FlattenedYAMLConfig(resolved_cfg)
        return cls(
            beta_start=yaml_config.value("diffusion", "beta_start", value_type=float, default=1e-4),
            beta_end=yaml_config.value("diffusion", "beta_end", value_type=float, default=0.02),
            num_steps=yaml_config.value("diffusion", "num_steps", value_type=int, default=100),
        )


try:
    DEFAULT_DIFFUSION_SCHEDULE = DiffusionSchedule.load_from_config()
except Exception:
    DEFAULT_DIFFUSION_SCHEDULE = DiffusionSchedule()

def linear_beta_schedule(
    schedule: DiffusionSchedule = DEFAULT_DIFFUSION_SCHEDULE,
) -> torch.Tensor:
    """Return the linear beta schedule for a diffusion schedule.

    Args:
        schedule: Diffusion schedule parameters.

    Returns:
        A tensor of shape ``(num_steps,)`` holding linearly spaced beta
        values from ``beta_start`` to ``beta_end`` inclusive.
    """
    if not isinstance(schedule, DiffusionSchedule):
        raise TypeError("schedule must be a DiffusionSchedule instance")
    if schedule.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if schedule.beta_start < 0.0 or schedule.beta_end < 0.0:
        raise ValueError("beta_start and beta_end must be non-negative")
    return torch.linspace(schedule.beta_start, schedule.beta_end, schedule.num_steps)
