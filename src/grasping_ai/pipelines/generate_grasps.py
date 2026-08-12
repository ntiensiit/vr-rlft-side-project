from pathlib import Path
from typing import cast

import numpy as np

from grasping_ai.inference.grasp_generator import GraspPoseGenerator


def build_generation_pipeline(
    object_point_cloud: np.ndarray,
    grasp_generator: GraspPoseGenerator,
    num_candidates: int,
) -> np.ndarray:
    """Run an end-to-end grasp-generation pipeline for a single object.

    Args:
        object_point_cloud: Object point cloud with shape ``(N, 3)``.
        grasp_generator: Callable grasp generator produced by inference.
        num_candidates: Number of candidate grasp poses to generate.

    Returns:
        Candidate grasp poses as a numpy array.
    """
    return grasp_generator(object_point_cloud, num_candidates)


def generate_grasps_for_dataset(
    dataset_point_clouds: list[np.ndarray],
    grasp_generator: GraspPoseGenerator,
    num_candidates: int,
) -> list[np.ndarray]:
    """Generate grasp candidates for a list of object point clouds.

    Args:
        dataset_point_clouds: List of per-object point clouds.
        grasp_generator: Callable grasp generator produced by inference.
        num_candidates: Number of candidate grasp poses to generate per object.

    Returns:
        List of per-object candidate grasp-pose arrays.
    """
    return [
        build_generation_pipeline(pc, grasp_generator, num_candidates)
        for pc in dataset_point_clouds
    ]


def write_generated_grasps(output_path: Path, grasps_by_object: dict[str, np.ndarray]) -> None:
    """Persist generated grasps to disk under the supplied output path.

    Args:
        output_path: Destination path for the generated-grasp file.
        grasps_by_object: Mapping from object identifier to grasp-pose array.
    """
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path instance")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from typing import Any
        np.save(output_path, cast(Any, grasps_by_object), allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to write generated grasps: {e}") from e
