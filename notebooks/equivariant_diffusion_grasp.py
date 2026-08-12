# OBSOLETE — this notebook predates the current supervised diffusion pipeline
# and references a non-existent ``data/raw/grasp_data`` directory plus an
# obsolete observation layout. Use ``scripts/train.py`` (diffusion) or
# ``scripts/train_flow.py`` (flow) together with ``scripts/prepare_data.py``
# (synthetic YCB dataset) instead.
#
# %%
!git clone https://github.com/ntiensiit/vr-rlft-side-project.git
%cd vr-rlft-side-project
!pip install -e .

# %%
from pathlib import Path
import torch
from grasping_ai.data.pointcloud_dataset import discover_dataset_files
from grasping_ai.data.transforms import save_grasp_dataset_index
from grasping_ai.inference.grasp_generator import (
    build_diffusion_grasp_generator,
    load_grasp_model_checkpoint,
)
from grasping_ai.pipelines.generate_grasps import (
    generate_grasps_for_dataset,
    write_generated_grasps,
)
from grasping_ai.pipelines.train import (
    build_supervised_training_components,
    run_training_pipeline,
)
from grasping_ai.sensors.pointcloud_sensor import (
    acquire_point_cloud_from_observation,
)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_root = Path("data/raw/grasp_data")
processed_root = Path("data/processed")
checkpoint_path = Path("artifacts/checkpoints/diffusion_model.pt")
output_path = Path("artifacts/exports/generated_grasps.npy")

# %%
records = discover_dataset_files(dataset_root)
entries = [{"path": str(record)} for record in records]
save_grasp_dataset_index(processed_root, entries)

# %%
components = build_supervised_training_components(
    feature_dim=128,
    hidden_dim=256,
    num_layers=4,
    learning_rate=0.0001,
    device=device,
)
print(components.keys())

# %%
run_training_pipeline(
    dataset_root=processed_root,
    checkpoint_path=checkpoint_path,
    feature_dim=128,
    hidden_dim=256,
    num_layers=4,
    learning_rate=0.0001,
    num_epochs=10,
    batch_size=32,
    device=device,
)

# %%
checkpoint = load_grasp_model_checkpoint(checkpoint_path, device=device)

# %%
generator = build_diffusion_grasp_generator(
    checkpoint, feature_dim=128, num_diffusion_steps=100, device=device
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
