from pathlib import Path

YcbObjectMesh = Path


def list_ycb_objects(ycb_root: Path) -> list[str]:
    """Enumerate available YCB object identifiers under a YCB root directory.

    Args:
        ycb_root: Root directory of the YCB object set.

    Returns:
        Sorted list of YCB object identifiers.
    """
    if not isinstance(ycb_root, Path):
        raise TypeError("ycb_root must be a pathlib.Path instance")
    if not ycb_root.is_dir():
        raise FileNotFoundError(f"YCB root directory '{ycb_root}' does not exist")

    objects = []
    for path in ycb_root.iterdir():
        if path.is_dir():
            objects.append(path.name)
    return sorted(objects)


def resolve_ycb_object_directory(ycb_root: Path, object_name: str) -> Path:
    """Resolve the on-disk directory of a YCB object.

    Args:
        ycb_root: Root directory of the YCB object set.
        object_name: Logical YCB object identifier such as ``"mustard_bottle"``.

    Returns:
        Path to the directory containing the YCB object assets.
    """
    if not isinstance(ycb_root, Path):
        raise TypeError("ycb_root must be a pathlib.Path instance")
    if not isinstance(object_name, str):
        raise TypeError("object_name must be a string")
    if not ycb_root.is_dir():
        raise FileNotFoundError(f"YCB root directory '{ycb_root}' does not exist")

    # 1. Check exact match
    direct_path = ycb_root / object_name
    if direct_path.is_dir():
        return direct_path

    # 2. Check suffix/prefix match
    for path in ycb_root.iterdir():
        if path.is_dir():
            if path.name == object_name:
                return path
            # Prefix match, e.g. "006_mustard_bottle" matching "mustard_bottle"
            if (
                path.name.endswith("_" + object_name)
                and len(path.name) > len(object_name) + 1
                and path.name[:3].isdigit()
            ):
                return path
            # Suffix match, e.g. "mustard_bottle" matching "006_mustard_bottle"
            if (
                object_name.endswith("_" + path.name)
                and len(object_name) > len(path.name) + 1
                and object_name[:3].isdigit()
            ):
                return path

    raise FileNotFoundError(f"YCB object '{object_name}' not found under '{ycb_root}'")


def find_ycb_mesh_file(object_dir: Path) -> YcbObjectMesh:
    """Locate the mesh file inside a YCB object directory.

    Args:
        object_dir: Directory of a single YCB object.

    Returns:
        Path to the mesh file (for example an OBJ) inside ``object_dir``.
    """
    if not isinstance(object_dir, Path):
        raise TypeError("object_dir must be a pathlib.Path instance")
    if not object_dir.is_dir():
        raise FileNotFoundError(f"YCB object directory '{object_dir}' does not exist")

    for path in object_dir.rglob("textured.obj"):
        if path.is_file():
            return path

    for path in object_dir.rglob("*.obj"):
        if path.is_file():
            return path

    for path in object_dir.rglob("*.ply"):
        if path.is_file():
            return path

    raise FileNotFoundError(f"No mesh file (.obj or .ply) found in '{object_dir}'")


def find_ycb_mjcf(object_dir: Path) -> Path:
    """Locate the MJCF XML description of a YCB object.

    This is the single discovery pattern for object MJCF files used across
    the grasp-simulation and RL-training pipelines.

    Args:
        object_dir: Directory of a single YCB object.

    Returns:
        Path to the object MJCF XML file inside ``object_dir``.

    Raises:
        TypeError: If ``object_dir`` is not a ``pathlib.Path``.
        FileNotFoundError: If no XML file exists under ``object_dir``.
    """
    if not isinstance(object_dir, Path):
        raise TypeError("object_dir must be a pathlib.Path instance")
    if not object_dir.is_dir():
        raise FileNotFoundError(f"YCB object directory '{object_dir}' does not exist")

    for xml_path in object_dir.glob("*.xml"):
        if xml_path.is_file():
            return xml_path
    for xml_path in object_dir.rglob("*.xml"):
        if xml_path.is_file():
            return xml_path

    raise FileNotFoundError(f"No MJCF XML file found in '{object_dir}'")


def ycb_object_exists(ycb_root: Path, object_name: str) -> bool:
    """Check whether a YCB object exists under the given root directory.

    Args:
        ycb_root: Root directory of the YCB object set.
        object_name: Logical YCB object identifier.

    Returns:
        ``True`` if the object is available, otherwise ``False``.
    """
    try:
        resolve_ycb_object_directory(ycb_root, object_name)
        return True
    except (FileNotFoundError, TypeError, ValueError):
        return False
