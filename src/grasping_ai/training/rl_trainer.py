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
    device_obj = torch.device(device)

    def step(transitions: list[RLTransition]) -> dict[str, float]:
        if not transitions:
            return {"loss": 0.0}

        observations = np.array([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rewards_arr = np.array([t[2] for t in transitions], dtype=np.float32)

        returns = compute_discounted_returns(transitions, 0.99)

        obs_t = torch.from_numpy(observations).float().to(device_obj)
        act_t = torch.from_numpy(actions).float().to(device_obj)
        ret_t = torch.from_numpy(returns).float().to(device_obj)

        policy.train()
        optimizer.zero_grad()

        pred_actions = policy(obs_t)
        action_diff = pred_actions - act_t
        log_prob = -0.5 * (action_diff ** 2).sum(dim=-1)

        advantages = ret_t - ret_t.mean()
        std = ret_t.std()
        if std > 1e-8:
            advantages = advantages / std

        clipped_advantages = torch.clamp(
            advantages, -clip_ratio, clip_ratio
        )
        policy_loss = -(log_prob * clipped_advantages).mean()
        entropy_bonus = -entropy_coefficient * log_prob.mean()
        loss = policy_loss + entropy_bonus

        loss.backward()
        optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "mean_reward": float(rewards_arr.mean()),
        }

    step.model = policy  # type: ignore[attr-defined]
    step.optimizer = optimizer  # type: ignore[attr-defined]
    return step


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
    if num_updates <= 0:
        raise ValueError("num_updates must be a positive integer")

    for _ in range(num_updates):
        try:
            rollout = next(rollout_iterator)
        except StopIteration:
            break

        update_step(rollout)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = getattr(update_step, "model", None)
    optimizer = getattr(update_step, "optimizer", None)
    checkpoint: dict[str, object] = {"epoch": num_updates}
    if model is not None:
        checkpoint["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, checkpoint_path)


def compute_discounted_returns(
    transitions: list[RLTransition], gamma: float
) -> np.ndarray:
    """Compute discounted returns for a list of RL transitions.

    Args:
        transitions: List of ``(obs, action, reward, next_obs, done)`` tuples.
        gamma: Discount factor in ``[0, 1]``.

    Returns:
        Discounted returns as a numpy array aligned with ``transitions``.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    n = len(transitions)
    returns = np.zeros(n, dtype=np.float64)
    running_return = 0.0
    for i in range(n - 1, -1, -1):
        reward = transitions[i][2]
        done = transitions[i][4]
        if done:
            running_return = 0.0
        running_return = reward + gamma * running_return
        returns[i] = running_return

    return returns.astype(np.float32)


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
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")

    n = len(transitions)
    advantages = np.zeros(n, dtype=np.float64)
    gae = 0.0

    for i in range(n - 1, -1, -1):
        obs, _action, reward, next_obs, done = transitions[i]
        v_current = value_fn(obs)
        v_next = 0.0 if done else value_fn(next_obs)
        delta = reward + gamma * v_next - v_current
        if done:
            gae = 0.0
        gae = delta + gamma * gae_lambda * gae
        advantages[i] = gae

    returns_arr = advantages + np.array(
        [value_fn(t[0]) for t in transitions], dtype=np.float64
    )
    return advantages.astype(np.float32), returns_arr.astype(np.float32)
