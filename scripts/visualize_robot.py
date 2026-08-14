from pathlib import Path

from grasping_ai.pipelines.visualize_robot import (
    load_visualization_scene,
    run_robot_viewer,
)

from grasping_ai.config.yaml_loader import optional_cli_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize the robot in an interactive MuJoCo viewer"
    )
    parser.add_argument("--robot-xml", type=Path, default=Path("deploy/robot.xml"))
    parser.add_argument("--object-id", type=str, default=None)
    parser.add_argument("--ycb-root", type=Path, default=None)
    parser.add_argument("--table-xml", type=optional_cli_path, default=Path("deploy/table.xml"))
    args = parser.parse_args()
    mj_model, mj_data = load_visualization_scene(
        args.robot_xml,
        object_id=args.object_id,
        ycb_root=args.ycb_root,
        table_xml_path=args.table_xml,
    )
    run_robot_viewer(mj_model, mj_data)
