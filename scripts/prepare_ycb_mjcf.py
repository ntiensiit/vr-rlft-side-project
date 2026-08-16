"""Convert YCB assets into MuJoCo MJCF scenes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from loguru import logger

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig
from grasping_ai.simulation.ycb import (
    find_ycb_mesh_file,
    list_ycb_objects,
)

YCB_ROOT = Path(str(FLATTENED_YAML_CONFIG.get("script.ycb_root", "data/raw/ycb")))

if TYPE_CHECKING:
    from omegaconf import DictConfig


def convert_ycb_to_mjcf(ycb_root: Path, output_root: Path) -> list[Path]:
    """Write MJCF object wrappers referencing the real YCB mesh meshes."""
    if not ycb_root.is_dir():
        msg = f"YCB root directory '{ycb_root}' does not exist"
        raise FileNotFoundError(msg)

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
            "    <worldbody>\n"
            f'        <body name="{object_id}" pos="0.5 0 0.3">\n'
            "            <freejoint/>\n"
            f'            <geom name="{object_id}_geom" type="mesh" mesh="{object_id}_mesh"/>\n'
            "        </body>\n"
            "    </worldbody>\n"
            "</mujoco>\n"
        )
        wrapper_path.write_text(wrapper_text, encoding="utf-8")
        generated.append(wrapper_path)
    return sorted(generated)

@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/prepare_ycb_mjcf")
def main(cfg: DictConfig) -> None:
    """Convert all YCB objects to MJCF wrappers and log the generated paths."""
    yaml_config = FlattenedYAMLConfig(cfg)
    generated = convert_ycb_to_mjcf(
        yaml_config.value("ycb_root", "paths", "ycb_root", value_type=Path, script_or=True, default=YCB_ROOT),
        yaml_config.value("ycb_mjcf", "paths", "ycb_mjcf", value_type=Path, script_or=True, required=True),
    )
    for path in generated:
        logger.info("{}", path)

if __name__ == "__main__":
    main()
