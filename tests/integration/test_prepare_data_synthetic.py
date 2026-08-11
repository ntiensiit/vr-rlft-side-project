import sys
from pathlib import Path

import open3d as o3d  # type: ignore[import-untyped]


def test_prepare_data_synthetic_pipeline(tmp_path):
    ycb_root = tmp_path / "ycb_raw"
    ycb_root.mkdir()

    obj_name = "006_mustard_bottle"
    obj_dir = ycb_root / obj_name
    obj_dir.mkdir()

    # Write a dummy textured.obj
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.1, depth=0.05)
    o3d.io.write_triangle_mesh(str(obj_dir / "textured.obj"), mesh)

    dataset_root = tmp_path / "dataset"
    output_index = tmp_path / "index.json"

    import subprocess
    cmd = [
        sys.executable,
        "scripts/prepare_data.py",
        "--mode", "synthetic",
        "--ycb-root", str(ycb_root),
        "--dataset-root", str(dataset_root),
        "--output-index", str(output_index),
        "--num-samples", "100",
        "--num-grasps", "5",
        "--seed", "42"
    ]

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")

    subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)

    # Verify .npy file exists
    npy_file = dataset_root / f"{obj_name}.npy"
    assert npy_file.is_file()

    # Verify index.json exists
    assert output_index.is_file()

    # Load and check the data structure using load_grasp_sample
    from grasping_ai.data.pointcloud_dataset import load_grasp_sample
    sample = load_grasp_sample(npy_file)
    assert "point_cloud" in sample
    assert "grasp_poses" in sample
    assert "scores" in sample
    assert "object_id" in sample
    assert sample["point_cloud"].shape == (100, 3)
