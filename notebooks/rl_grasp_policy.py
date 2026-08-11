# %%
!git clone https://github.com/ntiensiit/vr-rlft-side-project.git
%cd vr-rlft-side-project
!pip install -e .

# %%
from pathlib import Path
import torch
from grasping_ai.models.rl_policy import (
    build_policy_network,
    build_value_network,
)
from grasping_ai.pipelines.train_rl import (
    run_rl_training_pipeline,
)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
robot_xml_path = Path("deploy/robot.xml")
ycb_root = Path("data/raw/ycb")
policy_checkpoint_path = Path("artifacts/checkpoints/rl_policy.pt")

# %%
policy = build_policy_network(
    observation_dim=64, action_dim=7, hidden_dim=256, num_layers=2
)
value_net = build_value_network(
    observation_dim=64, hidden_dim=256, num_layers=2
)
print(policy, value_net)

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
