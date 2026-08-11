from pathlib import Path

from grasping_ai.pipelines.train_rl import run_rl_training_pipeline


def train_rl_main(
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
    """Run the RL training pipeline against a MuJoCo+YCB environment.

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
    run_rl_training_pipeline(
        robot_xml_path=robot_xml_path,
        ycb_root=ycb_root,
        object_ids=object_ids,
        policy_checkpoint_path=policy_checkpoint_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        num_updates=num_updates,
        gamma=gamma,
        device=device,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an RL grasping policy")
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--ycb-root", type=Path, required=True)
    parser.add_argument("--object-ids", type=str, nargs="+", required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--observation-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--num-updates", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    train_rl_main(
        args.robot_xml,
        args.ycb_root,
        args.object_ids,
        args.policy_checkpoint,
        args.observation_dim,
        args.action_dim,
        args.hidden_dim,
        args.learning_rate,
        args.num_updates,
        args.gamma,
        args.device,
    )
