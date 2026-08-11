# %%
!git clone https://github.com/ntiensiit/vr-rlft-side-project.git
%cd vr-rlft-side-project
!pip install -e .

# %%
from pathlib import Path
import numpy as np
import torch
from grasping_ai.models.rl_policy import (
    build_policy_network,
    build_value_network,
)
from grasping_ai.pipelines.train_rl import (
    build_rl_environment,
    collect_rl_rollout,
    run_rl_training_pipeline,
)
from grasping_ai.training.rl_trainer import build_rl_training_step

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
robot_xml_path = Path("deploy/robot.xml")
ycb_root = Path("data/raw/ycb")
policy_checkpoint_path = Path("artifacts/checkpoints/rl_policy.pt")

# %%
env_state, step_fn, contact_reporter = build_rl_environment(
    robot_xml_path=robot_xml_path,
    ycb_root=ycb_root,
    object_id="003_cracker_box",
    observation_dim=64,
    action_dim=7,
)

# %%
policy = build_policy_network(
    observation_dim=64, action_dim=7, hidden_dim=256, num_layers=2
)
value_net = build_value_network(
    observation_dim=64, hidden_dim=256, num_layers=2
)
print(policy, value_net)

# %%
optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)

# %%
update_step = build_rl_training_step(
    policy=policy,
    optimizer=optimizer,
    clip_ratio=0.2,
    entropy_coefficient=0.01,
    device=device,
)
print(update_step)

# %%
def policy_runner(_obs):
    return np.zeros(7)

# %%
rollout = collect_rl_rollout(env_state, policy_runner, num_steps=100)
print(len(rollout))

# %%
run_rl_training_pipeline(
    robot_xml_path=robot_xml_path,
    ycb_root=ycb_root,
    object_ids=["003_cracker_box"],
    policy_checkpoint_path=policy_checkpoint_path,
    observation_dim=64,
    action_dim=7,
    hidden_dim=256,
    learning_rate=0.0003,
    num_updates=10,
    gamma=0.99,
    device=device,
)
