# %%
!git clone https://github.com/ntiensiit/vr-rlft-side-project.git
%cd vr-rlft-side-project
!pip install -e .

# %%
from pathlib import Path
import torch
from grasping_ai.data.pointcloud_dataset import discover_dataset_files
from grasping_ai.inference.grasp_generator import (
    build_flow_grasp_generator,
    load_grasp_model_checkpoint,
)
from grasping_ai.models.flow import build_flow_field
from grasping_ai.pipelines.generate_grasps import (
    generate_grasps_for_dataset,
    write_generated_grasps,
)
from grasping_ai.sensors.pointcloud_sensor import (
    acquire_point_cloud_from_observation,
)
from grasping_ai.training.losses import build_flow_matching_loss
from grasping_ai.training.trainer import (
    build_adam_optimizer,
    build_training_step,
    run_training_loop,
)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_root = Path("data/raw/grasp_data")
processed_root = Path("data/processed")
checkpoint_path = Path("artifacts/checkpoints/flow_model.pt")
output_path = Path("artifacts/exports/generated_grasps_flow.npy")

# %%
records = discover_dataset_files(dataset_root)
print(len(records))

# %%
flow_field = build_flow_field(feature_dim=128, hidden_dim=256, num_layers=4)
loss_fn = build_flow_matching_loss(flow_field)
optimizer = build_adam_optimizer(flow_field.parameters(), learning_rate=0.0001)

# %%
training_step = build_training_step(
    model=flow_field,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
)

# %%
dataloader = []
run_training_loop(
    training_step=training_step,
    dataloader=dataloader,
    num_epochs=10,
    checkpoint_path=checkpoint_path,
    log_every=10,
)

# %%
checkpoint = load_grasp_model_checkpoint(checkpoint_path, device=device)

# %%
generator = build_flow_grasp_generator(
    checkpoint, feature_dim=128, num_flow_steps=50, device=device
)

# %%
obs_paths = [processed_root / "obs_0.npy", processed_root / "obs_1.npy"]
point_clouds = [acquire_point_cloud_from_observation(p) for p in obs_paths]

# %%
grasps = generate_grasps_for_dataset(point_clouds, generator, num_candidates=10)

# %%
write_generated_grasps(
    output_path, {f"object_{i}": grasp for i, grasp in enumerate(grasps)}
)
