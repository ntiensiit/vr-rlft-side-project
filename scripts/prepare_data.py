from pathlib import Path

from grasping_ai.data.pointcloud_dataset import discover_dataset_files
from grasping_ai.data.transforms import save_grasp_dataset_index


def prepare_data(dataset_root: Path, output_index_path: Path) -> None:
    """Discover dataset files and write a dataset index file.

    Args:
        dataset_root: Root directory containing raw dataset records.
        output_index_path: Destination path for the generated index file.
    """
    records = discover_dataset_files(dataset_root)
    entries = [{"path": str(record)} for record in records]
    save_grasp_dataset_index(output_index_path.parent, entries)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare grasp dataset index")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    args = parser.parse_args()
    prepare_data(args.dataset_root, args.output_index)
