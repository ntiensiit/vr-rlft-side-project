import argparse
from pathlib import Path

MJCF_TEMPLATE = """\
<mujoco model="{object_id}">
    <compiler angle="radian"/>
    <asset><mesh name="{object_id}_mesh" file="{mesh_path}"/></asset>
    <worldbody>
        <body name="{object_id}" pos="0 0 0.05">
            <freejoint/>
            <geom name="{object_id}_geom" type="mesh" mesh="{object_id}_mesh"/>
        </body>
    </worldbody>
</mujoco>
"""


def convert_ycb_to_mjcf(ycb_root: Path, output_root: Path) -> list[Path]:
    """Write MJCF object wrappers referencing the real YCB mesh meshes.

    The shipped YCB objects embed OpenRAVE ``<KinBody/>`` descriptions rather
    than MuJoCo MJCF, so a thin MJCF wrapper is written per object that loads
    the real mesh file. The wrappers are consumed by the grasp-simulation and
    RL-training pipelines through the normal ``resolve_ycb_object_directory``
    and ``find_ycb_mjcf`` discovery path.

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
        mesh_path = mesh.resolve().as_posix()
        object_out_dir = output_root / object_id
        object_out_dir.mkdir(parents=True, exist_ok=True)
        wrapper_path = object_out_dir / "object.xml"
        wrapper_path.write_text(
            MJCF_TEMPLATE.format(object_id=object_id, mesh_path=mesh_path),
            encoding="utf-8",
        )
        generated.append(wrapper_path)
    return sorted(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw YCB objects to MJCF wrappers for simulation"
    )
    parser.add_argument("--ycb-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    generated = convert_ycb_to_mjcf(args.ycb_root, args.output_root)
    for path in generated:
        print(path)
