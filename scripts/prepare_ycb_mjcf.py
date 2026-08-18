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
# Measured masses from the YCB Object and Model Set table (Calli et al.).
# Supplying them prevents MuJoCo from inferring mass from its 1000 kg/m^3
# default mesh density (which makes the hollow cracker box weigh 2.17 kg).
# Keep every curated ``objects.ids`` entry here: dynamics validation must not
# silently use mesh volume as a proxy for a lightweight cup or cardboard box.
YCB_MASSES_KG = {
    "003_cracker_box": 0.411,
    "004_sugar_box": 0.514,
    "005_tomato_soup_can": 0.349,
    "006_mustard_bottle": 0.603,
    "009_gelatin_box": 0.097,
    "010_potted_meat_can": 0.370,
    "036_wood_block": 0.729,
    "065-a_cups": 0.020,
    "065-b_cups": 0.020,
    "065-c_cups": 0.020,
    "065-d_cups": 0.020,
    "065-g_cups": 0.020,
    "065-i_cups": 0.020,
    "077_rubiks_cube": 0.094,
}

if TYPE_CHECKING:
    from omegaconf import DictConfig


def convert_ycb_to_mjcf(
    ycb_root: Path,
    output_root: Path,
    *,
    object_ids: list[str] | None = None,
) -> list[Path]:
    """Write MJCF wrappers for selected or all YCB meshes.

    ``None`` retains the library helper's all-object behavior.  CLI callers
    supply the curated ``objects.ids`` list, so preparing project assets never
    silently expands to every downloaded YCB object.
    """
    if not ycb_root.is_dir():
        msg = f"YCB root directory '{ycb_root}' does not exist"
        raise FileNotFoundError(msg)

    if object_ids is not None and not object_ids:
        raise ValueError("object_ids must not be empty when explicitly provided")

    generated: list[Path] = []
    selected_object_ids = list(dict.fromkeys(object_ids)) if object_ids is not None else list_ycb_objects(ycb_root)

    # Resolve every mesh before writing anything.  A typo or a missing raw
    # asset must fail as one transaction rather than leave a partial MJCF set.
    selected_meshes = {object_id: find_ycb_mesh_file(ycb_root / object_id) for object_id in selected_object_ids}

    output_root.mkdir(parents=True, exist_ok=True)
    for object_id in selected_object_ids:
        mesh = selected_meshes[object_id]
        object_out_dir = output_root / object_id
        object_out_dir.mkdir(parents=True, exist_ok=True)
        wrapper_path = object_out_dir / "object.xml"
        mesh_dir = mesh.parent.resolve().as_posix()
        mass_attribute = f' mass="{YCB_MASSES_KG[object_id]}"' if object_id in YCB_MASSES_KG else ""
        wrapper_text = (
            f'<mujoco model="{object_id}">\n'
            '    <compiler angle="radian"/>\n'
            f'    <asset><mesh name="{object_id}_mesh" file="{mesh_dir}/{mesh.name}"/></asset>\n'
            "    <worldbody>\n"
            f'        <body name="{object_id}" pos="0.5 0 0.28">\n'
            "            <freejoint/>\n"
            f'            <geom name="{object_id}_geom" type="mesh" mesh="{object_id}_mesh"{mass_attribute}/>\n'
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
        object_ids=yaml_config.value(
            "object_ids",
            "objects",
            "ids",
            value_type=list[str],
            script_or=True,
            default=None,
        ),
    )
    for path in generated:
        logger.info("{}", path)


if __name__ == "__main__":
    main()
