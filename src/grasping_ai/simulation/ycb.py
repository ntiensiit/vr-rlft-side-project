from pathlib import Path

YcbObjectMesh = Path


def list_ycb_objects(ycb_root: Path) -> list[str]:
    """Enumerate available YCB object identifiers under a YCB root directory.

    Args:
        ycb_root: Root directory of the YCB object set.

    Returns:
        Sorted list of YCB object identifiers.
    """
    raise NotImplementedError


def resolve_ycb_object_directory(ycb_root: Path, object_name: str) -> Path:
    """Resolve the on-disk directory of a YCB object.

    Args:
        ycb_root: Root directory of the YCB object set.
        object_name: Logical YCB object identifier such as ``"mustard_bottle"``.

    Returns:
        Path to the directory containing the YCB object assets.
    """
    raise NotImplementedError


def find_ycb_mesh_file(object_dir: Path) -> YcbObjectMesh:
    """Locate the mesh file inside a YCB object directory.

    Args:
        object_dir: Directory of a single YCB object.

    Returns:
        Path to the mesh file (for example an OBJ) inside ``object_dir``.
    """
    raise NotImplementedError


def ycb_object_exists(ycb_root: Path, object_name: str) -> bool:
    """Check whether a YCB object exists under the given root directory.

    Args:
        ycb_root: Root directory of the YCB object set.
        object_name: Logical YCB object identifier.

    Returns:
        ``True`` if the object is available, otherwise ``False``.
    """
    raise NotImplementedError
