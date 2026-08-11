import json
from pathlib import Path

import numpy as np

from grasping_ai.pipelines.simulate_grasp import run_simulation_sweep


def run_simulation_main(
    grasps_path: Path,
    object_id: str,
    ycb_root: Path,
    robot_xml_path: Path,
    table_xml_path: Path | None,
    output_path: Path,
    num_simulation_steps: int,
    gripper_close_command: list[float],
) -> None:
    """Load grasps from disk and execute them in MuJoCo for a YCB object.

    Args:
        grasps_path: Path to a file of grasp poses to execute.
        object_id: Logical YCB object identifier to load.
        ycb_root: Root directory of the YCB object set.
        robot_xml_path: Path to the robot MJCF description.
        table_xml_path: Optional path to a workbench/table MJCF description.
        output_path: Destination path for the simulation outcomes.
        num_simulation_steps: Number of physics steps per grasp attempt.
        gripper_close_command: Gripper command used to close the gripper.
    """
    grasp_poses = np.load(grasps_path)
    outcomes = run_simulation_sweep(
        grasp_poses=grasp_poses,
        object_id=object_id,
        ycb_root=ycb_root,
        robot_xml_path=robot_xml_path,
        table_xml_path=table_xml_path,
        num_simulation_steps=num_simulation_steps,
        gripper_close_command=np.asarray(gripper_close_command, dtype=np.float64),
    )
    with output_path.open("w") as fp:
        json.dump(_serialize_outcomes(outcomes), fp)


def _serialize_outcomes(outcomes: object) -> list[dict[str, object]]:
    """Convert numpy-valued simulation outcomes to JSON-serializable forms.

    Args:
        outcomes: List of outcome dictionaries produced by ``run_simulation_sweep``.

    Returns:
        A list of outcome dictionaries with numpy arrays replaced by lists.
    """
    serialized: list[dict[str, object]] = []
    for outcome in outcomes:  # type: ignore[union-attr, attr-defined]
        converted: dict[str, object] = {}
        for key, value in outcome.items():  # type: ignore[union-attr]
            converted[key] = value.tolist() if hasattr(value, "tolist") else value
        serialized.append(converted)
    return serialized


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run grasps in MuJoCo simulation")
    parser.add_argument("--grasps", type=Path, required=True)
    parser.add_argument("--object-id", type=str, required=True)
    parser.add_argument("--ycb-root", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--table-xml", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-simulation-steps", type=int, required=True)
    parser.add_argument("--gripper-close-command", type=float, nargs="+", required=True)
    args = parser.parse_args()
    run_simulation_main(
        args.grasps,
        args.object_id,
        args.ycb_root,
        args.robot_xml,
        args.table_xml,
        args.output,
        args.num_simulation_steps,
        args.gripper_close_command,
    )
