"""Download YCB object assets used by simulation and data prep."""

from __future__ import annotations

import glob
import json
import shutil
import tarfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import hydra
from loguru import logger
from omegaconf import DictConfig

from grasping_ai.config.config import (
    SCRIPTS_CONFIG_PATH,
    config_value,
)

YCB_BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
USER_AGENT = "Mozilla/5.0"
DOWNLOAD_BLOCK_SIZE = 65536


def fetch_objects(url: str) -> list[str]:
    """Fetch the object list from the YCB benchmark index URL."""
    with urlopen(url) as response:
        payload = json.loads(response.read())
    objects = payload["objects"]
    if not isinstance(objects, list):
        msg = "YCB object index did not contain a list of objects"
        raise TypeError(msg)
    return [str(object_id) for object_id in objects]


def download_file(url: str, filename: Path, max_retries: int = 5) -> None:
    """Download a file from ``url`` with retries."""
    for attempt in range(max_retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as remote_file:
                file_size_header = remote_file.getheader("Content-Length")
                file_size = int(file_size_header) if file_size_header else 0
                logger.info("Downloading: {} ({:.2f} MB)", filename, file_size / 1_000_000.0)

                file_size_dl = 0
                with filename.open("wb") as local_file:
                    while True:
                        buffer = remote_file.read(DOWNLOAD_BLOCK_SIZE)
                        if not buffer:
                            break

                        file_size_dl += len(buffer)
                        local_file.write(buffer)
                        if file_size > 0:
                            status = (
                                f"{int(file_size_dl / 1_000_000.0):10d}  "
                                f"[{file_size_dl * 100.0 / file_size:3.2f}%]"
                            )
                        else:
                            status = f"{int(file_size_dl / 1_000_000.0):10d}"
                        status = status + chr(8) * (len(status) + 1)
                        logger.info("{}", status)
        except (OSError, URLError) as exc:
            logger.warning("Error downloading (attempt {}/{}): {}", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
        else:
            return


def tgz_url(ycb_object: str, file_type: str) -> str:
    """Return the TGZ file URL for a YCB object and dataset type."""
    if file_type in {"berkeley_rgbd", "berkeley_rgb_highres"}:
        return f"{YCB_BASE_URL}berkeley/{ycb_object}/{ycb_object}_{file_type}.tgz"
    if file_type == "berkeley_processed":
        return f"{YCB_BASE_URL}berkeley/{ycb_object}/{ycb_object}_berkeley_meshes.tgz"
    return f"{YCB_BASE_URL}google/{ycb_object}_{file_type}.tgz"


def extract_tgz(filename: Path, output_dir: Path) -> None:
    """Extract a TGZ archive into ``output_dir``."""
    with tarfile.open(filename, "r:gz") as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(path=output_dir, filter="data")
        else:
            archive.extractall(path=output_dir)

    for _ in range(10):
        try:
            filename.unlink()
            break
        except OSError:
            time.sleep(1)


def check_url(url: str) -> bool:
    """Return whether ``url`` responds to a HEAD request."""
    try:
        request = Request(url)
        request.get_method = lambda: "HEAD"
        with urlopen(request):
            pass
    except (OSError, URLError):
        return False
    else:
        return True


def _is_already_extracted(object_dir: Path, file_type: str) -> bool:
    if file_type == "google_16k":
        return (object_dir / "google_16k").exists()
    if file_type == "berkeley_processed":
        return (object_dir / "clouds").exists()
    return False


def _cleanup_failed_object(output_directory: Path, object_id: str) -> None:
    object_dir = output_directory / object_id
    if object_dir.exists():
        shutil.rmtree(object_dir, ignore_errors=True)

    tgz_pattern = output_directory / f"{object_id}_*.tgz"
    for file_name in glob.glob(str(tgz_pattern)):
        for _ in range(5):
            try:
                Path(file_name).unlink()
                break
            except OSError:
                time.sleep(1)


def _process_object(
    object_id: str,
    output_directory: Path,
    files_to_download: list[str],
    extract: bool,
) -> None:
    object_dir = output_directory / object_id
    for file_type in files_to_download:
        if extract:
            if _is_already_extracted(object_dir, file_type):
                logger.info("Skipping {} {}: already extracted.", object_id, file_type)
                continue
        else:
            filename = output_directory / f"{object_id}_{file_type}.tgz"
            if filename.is_file():
                logger.info("Skipping {} {}: already downloaded.", object_id, file_type)
                continue

        url = tgz_url(object_id, file_type)
        if not check_url(url):
            continue

        filename = output_directory / f"{object_id}_{file_type}.tgz"
        download_file(url, filename)
        if extract:
            extract_tgz(filename, output_directory)


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/download_ycb_dataset")
def main(cfg: DictConfig) -> None:
    output_directory = config_value(
        cfg, "output_directory", "download", "output_directory", value_type=Path, script_or=True
    )
    objects_to_download = config_value(cfg, "objects_to_download", "download", "objects", value_type=list[str], script_or=True)
    files_to_download = config_value(cfg, "files_to_download", "download", "files", value_type=list[str], script_or=True)
    extract = config_value(cfg, "extract", "download", "extract", value_type=bool, default=True, script_or=True)
    objects_url = str(config_value(cfg, "objects_url", "download", "objects_url", value_type=object, script_or=True))

    output_directory.mkdir(parents=True, exist_ok=True)
    objects = fetch_objects(objects_url)

    for object_id in objects:
        if object_id not in objects_to_download:
            continue

        try:
            _process_object(object_id, output_directory, files_to_download, extract)
        except (OSError, URLError) as exc:
            logger.error("Failed to process {}: {}. Cleaning up...", object_id, exc)
            _cleanup_failed_object(output_directory, object_id)


if __name__ == "__main__":
    main()
