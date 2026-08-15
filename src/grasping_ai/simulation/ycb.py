from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loguru import logger
from theseus import Node, default_tokenizer  # type: ignore[import-untyped]

from grasping_ai.utils.path_validation import require_path

YcbObjectMesh = Path


def tokenize_ycb_object_name(object_name: str) -> list[str]:
    """Tokenize a YCB object identifier for vocabulary-based alias matching.

    Uses the ``theseus`` dependency's ``default_tokenizer`` so object names
    with spaces, underscores, or numeric YCB prefixes resolve consistently.

    Args:
        object_name: Logical object identifier such as ``"mustard_bottle"`` or
            ``"006 mustard bottle"``.

    Returns:
        Lowercased word tokens extracted from ``object_name``.
    """
    if not isinstance(object_name, str):
        raise TypeError("object_name must be a string")

    normalized = object_name.replace("-", " ").replace("_", " ")
    return default_tokenizer(normalized)


def build_ycb_object_name_classifier(
    ycb_root: Path,
) -> Callable[[str], str | None]:
    """Build a vocabulary classifier that maps alias strings to YCB directory names.

    Profiles each installed YCB object directory with ``theseus.Node`` and
    returns the best-matching canonical directory name for free-form queries
    such as ``"mustard bottle"`` or ``"006 mustard bottle"``.

    Args:
        ycb_root: Root directory of the YCB object set.

    Returns:
        Callable that maps a query string to a directory name, or ``None`` when
        no object profile matches.
    """
    require_path(ycb_root, "ycb_root")
    if not ycb_root.is_dir():
        raise FileNotFoundError(f"YCB root directory '{ycb_root}' does not exist")

    object_names = list_ycb_objects(ycb_root)
    if not object_names:
        raise ValueError(f"No YCB objects found under '{ycb_root}'")

    vocabularies: dict[str, set[str]] = {}
    for name in object_names:
        node = Node(documents=[tokenize_ycb_object_name(name)], name=name)
        vocabularies[name] = set(node.counter.keys())

    def classify(query: str) -> str | None:
        query_tokens = set(tokenize_ycb_object_name(query))
        if not query_tokens:
            return None

        best_name: str | None = None
        best_hits = 0
        for name, vocabulary in vocabularies.items():
            hits = len(query_tokens & vocabulary)
            if hits > best_hits:
                best_hits = hits
                best_name = name

        if best_hits == 0 or best_name is None:
            return None

        # Reject weak partial matches (e.g. a single shared token across objects).
        if best_hits < len(query_tokens):
            return None
        return best_name

    return classify


def list_ycb_objects(ycb_root: Path) -> list[str]:
    """Enumerate available YCB object identifiers under a YCB root directory.

    Args:
        ycb_root: Root directory of the YCB object set.

    Returns:
        Sorted list of YCB object identifiers.
    """
    require_path(ycb_root, "ycb_root")
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
    require_path(ycb_root, "ycb_root")
    if not isinstance(object_name, str):
        raise TypeError("object_name must be a string")
    if not ycb_root.is_dir():
        raise FileNotFoundError(f"YCB root directory '{ycb_root}' does not exist")


    # 1. Check exact match
    direct_path = ycb_root / object_name
    if direct_path.is_dir():
        logger.info("Resolved YCB object '{}' directly to directory: {}", object_name, direct_path)
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

    matched = build_ycb_object_name_classifier(ycb_root)(object_name)
    if matched is not None:
        resolved = ycb_root / matched
        logger.info("Resolved YCB object '{}' via classifier to directory: {}", object_name, resolved)
        return resolved

    raise FileNotFoundError(f"YCB object '{object_name}' not found under '{ycb_root}'")


def find_ycb_mesh_file(object_dir: Path) -> YcbObjectMesh:
    """Locate the mesh file inside a YCB object directory.

    Args:
        object_dir: Directory of a single YCB object.

    Returns:
        Path to the mesh file (for example an OBJ) inside ``object_dir``.
    """
    require_path(object_dir, "object_dir")
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
    require_path(object_dir, "object_dir")
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
