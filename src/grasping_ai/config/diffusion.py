from dataclasses import dataclass

import torch


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
