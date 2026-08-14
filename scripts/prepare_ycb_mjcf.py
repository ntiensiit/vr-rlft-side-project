import argparse
from pathlib import Path

from grasping_ai.config.yaml_loader import (
    config_path,
    load_project_yaml_config,
    parse_config_dir_from_argv,
)


def convert_ycb_to_mjcf(ycb_root: Path, output_root: Path) -> list[Path]:
    """Write MJCF object wrappers referencing the real YCB mesh meshes.

    The shipped YCB objects embed OpenRAVE ``<KinBody/>`` descriptions rather
    than MuJoCo MJCF, so a thin MJCF wrapper is written per object that loads
    the real mesh file. The wrappers are consumed by the grasp-simulation and
    RL-training pipelines through the normal ``resolve_ycb_object_directory``
    and ``find_ycb_mjcf`` discovery path.

    Mesh references use the mesh's absolute path via MuJoCo's ``meshdir``
    because MuJoCo resolves ``<include>``d-file meshdir relative to the
    including scene rather than the included file, which prevents reliable
    relocatable relative paths. The wrappers are therefore regenerable from
    source (``python scripts/prepare_ycb_mjcf.py``) for any fresh checkout
    rather than being relocatable as-is.

    Args:
        ycb_root: Root directory of the raw YCB object set.
        output_root: Destination directory for the MJCF wrapper set.

    Returns:
        Sorted list of generated MJCF wrapper paths.
    """
    from grasping_ai.simulation.ycb import find_ycb_mesh_file, list_ycb_objects

    if not isinstance(ycb_root, Path):
        raise TypeError("ycb_root must be a pathlib.Path instance")
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path instance")
    if not ycb_root.is_dir():
        raise FileNotFoundError(f"YCB root directory '{ycb_root}' does not exist")

    output_root.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for object_id in list_ycb_objects(ycb_root):
        object_dir = ycb_root / object_id
        mesh = find_ycb_mesh_file(object_dir)
        object_out_dir = output_root / object_id
        object_out_dir.mkdir(parents=True, exist_ok=True)
        wrapper_path = object_out_dir / "object.xml"
        mesh_dir = mesh.parent.resolve().as_posix()
        wrapper_text = (
            f'<mujoco model="{object_id}">\n'
            '    <compiler angle="radian"/>\n'
            f'    <asset><mesh name="{object_id}_mesh" file="{mesh_dir}/{mesh.name}"/></asset>\n'
            '    <worldbody>\n'
            f'        <body name="{object_id}" pos="0.5 0 0.1">\n'
            '            <freejoint/>\n'
            f'            <geom name="{object_id}_geom" type="mesh" mesh="{object_id}_mesh"/>\n'
            "        </body>\n"
            "    </worldbody>\n"
            "</mujoco>\n"
        )
        wrapper_path.write_text(wrapper_text, encoding="utf-8")
        generated.append(wrapper_path)
    return sorted(generated)


if __name__ == "__main__":
    config_dir = parse_config_dir_from_argv()
    cfg = load_project_yaml_config(config_dir, "base", "data")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-dir", type=Path, default=config_dir)
    parser = argparse.ArgumentParser(
        description="Convert raw YCB objects to MJCF wrappers for simulation",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--ycb-root",
        type=Path,
        default=config_path(cfg, "paths", "ycb_root"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config_path(cfg, "paths", "ycb_mjcf"),
    )
    args = parser.parse_args()
    if args.ycb_root is None:
        parser.error(
            "--ycb-root is required (set in configs/data/default.yaml paths.ycb_root "
            "or pass explicitly)"
        )
    if args.output_root is None:
        parser.error(
            "--output-root is required (set in configs/base.yaml paths.ycb_mjcf "
            "or pass explicitly)"
        )
    generated = convert_ycb_to_mjcf(args.ycb_root, args.output_root)
    for path in generated:
        print(path)
