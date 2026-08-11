from pathlib import Path

from grasping_ai.simulation.mujoco_env import ContactReporter, SimulationStep


def run_rl_training_pipeline(
    robot_xml_path: Path,
    ycb_root: Path,
    object_ids: list[str],
    policy_checkpoint_path: Path,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    learning_rate: float,
    num_updates: int,
    gamma: float,
    device: str,
) -> None:
    """Run an end-to-end RL training pipeline using MuJoCo as the environment.

    Args:
        robot_xml_path: Path to the robot MJCF description used in training.
        ycb_root: Root directory of the YCB object set.
        object_ids: YCB object identifiers used during training rollouts.
        policy_checkpoint_path: Destination path for the trained policy.
        observation_dim: Dimensionality of the policy observation vector.
        action_dim: Dimensionality of the policy action vector.
        hidden_dim: Hidden width of the policy and value networks.
        learning_rate: Learning rate for the policy optimizer.
        num_updates: Number of policy update steps to perform.
        gamma: Discount factor for return computation.
        device: Device identifier such as ``"cpu"`` or ``"cuda"``.
    """
    raise NotImplementedError


def build_rl_environment(
    robot_xml_path: Path,
    ycb_root: Path,
    object_id: str,
    observation_dim: int,
    action_dim: int,
) -> tuple[object, SimulationStep, ContactReporter]:
    """Construct a closed-loop RL environment over a MuJoCo scene.

    Args:
        robot_xml_path: Path to the robot MJCF description.
        ycb_root: Root directory of the YCB object set.
        object_id: YCB object identifier used as the manipulation target.
        observation_dim: Dimensionality of observations produced by the env.
        action_dim: Dimensionality of actions accepted by the env.

    Returns:
        A tuple ``(env_state, step, contacts)`` providing a stepping interface
        over the constructed RL environment.
    """
    raise NotImplementedError


def collect_rl_rollout(
    env_state: object,
    policy_runner: object,
    num_steps: int,
) -> list[tuple[object, object, float, object, bool]]:
    """Collect a rollout of environment transitions under the supplied policy.

    Args:
        env_state: Opaque environment state handle.
        policy_runner: Callable returning an action given an observation.
        num_steps: Number of environment steps to collect.

    Returns:
        A list of ``(obs, action, reward, next_obs, done)`` transition tuples.
    """
    raise NotImplementedError
