from pathlib import Path

import numpy as np
import pytest

from grasping_ai.inference.policy_runner import (
    build_rl_policy_runner,
    load_rl_policy_checkpoint,
)
from grasping_ai.pipelines.train_rl import run_rl_training_pipeline

MINIMAL_ACTUATED_ROBOT_XML = """\
<mujoco model="minimal_actuated_robot">
    <compiler angle="radian"/>
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="box" size="0.1 0.1 0.1"/>
            <body name="link1" pos="0 0 0.2">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" limited="true"/>
                <geom name="link1_geom" type="cylinder" size="0.05 0.1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor1" joint="joint1" gear="1" ctrlrange="-1.0 1.0"/>
    </actuator>
</mujoco>
"""

@pytest.fixture
def robot_xml(tmp_path: Path) -> Path:
    path = tmp_path / "robot.xml"
    path.write_text(MINIMAL_ACTUATED_ROBOT_XML, encoding="utf-8")
    return path

def test_sb3_training_and_inference_compatibility(robot_xml, tmp_path):
    checkpoint_path = tmp_path / "policy.pt"
    log_dir = tmp_path / "tb_logs"

    # Run the pipeline for a tiny number of updates (e.g. 1 update = 64 steps)
    run_rl_training_pipeline(
        robot_xml_path=robot_xml,
        ycb_root=tmp_path,  # unused since object_ids is empty
        object_ids=[],
        policy_checkpoint_path=checkpoint_path,
        observation_dim=2,  # qpos (1) + qvel (1)
        action_dim=1,
        hidden_dim=16,
        learning_rate=1e-3,
        num_updates=1,
        gamma=0.99,
        device="cpu",
        seed=42,
        experiment_log_dir=log_dir,
    )

    assert checkpoint_path.is_file()

    # Load and build inference policy runner
    checkpoint = load_rl_policy_checkpoint(checkpoint_path, "cpu")
    assert "model_state_dict" in checkpoint

    runner = build_rl_policy_runner(
        checkpoint=checkpoint,
        observation_dim=2,
        action_dim=1,
        device="cpu",
    )

    obs = np.array([0.5, -0.2], dtype=np.float32)
    action = runner(obs)

    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert np.isfinite(action).all()
