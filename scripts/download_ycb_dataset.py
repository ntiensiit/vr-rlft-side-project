"""Download YCB object assets used by simulation and data prep."""

from __future__ import annotations

import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import Request, urlopen

import hydra
from loguru import logger

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH

if TYPE_CHECKING:
    from omegaconf import DictConfig

OUTPUT_DIRECTORY = Path(str(FLATTENED_YAML_CONFIG.get("download.output_directory")))
OBJECTS_TO_DOWNLOAD = [str(object_id) for object_id in FLATTENED_YAML_CONFIG.get("download.objects", [])]
FILES_TO_DOWNLOAD = [str(file_type) for file_type in FLATTENED_YAML_CONFIG.get("download.files", [])]
EXTRACT = bool(FLATTENED_YAML_CONFIG.get("download.extract", True))
OBJECTS_URL = str(FLATTENED_YAML_CONFIG.get("download.objects_url"))
BASE_URL = str(FLATTENED_YAML_CONFIG.get("download.base_url"))
USER_AGENT = str(FLATTENED_YAML_CONFIG.get("download.user_agent"))
BLOCK_SIZE = int(FLATTENED_YAML_CONFIG.get("download.block_size", 65536))
MAX_RETRIES = int(FLATTENED_YAML_CONFIG.get("download.max_retries", 5))
TIMEOUT_SECONDS = int(FLATTENED_YAML_CONFIG.get("download.timeout_seconds", 30))
RETRY_SLEEP_SECONDS = float(FLATTENED_YAML_CONFIG.get("download.retry_sleep_seconds", 2))
UNPACK_DELETE_RETRIES = int(FLATTENED_YAML_CONFIG.get("download.unpack_delete_retries", 10))
CLEANUP_DELETE_RETRIES = int(FLATTENED_YAML_CONFIG.get("download.cleanup_delete_retries", 5))
CLEANUP_SLEEP_SECONDS = float(FLATTENED_YAML_CONFIG.get("download.cleanup_sleep_seconds", 1))
BERKELEY_RGB_TYPES = tuple(FLATTENED_YAML_CONFIG.get("download.berkeley_rgb_types", []))


def _validate_https_url(url: str) -> None:
    """Raise ``ValueError`` unless ``url`` uses the HTTPS scheme."""
    if not url.startswith("https://"):
        msg = f"Refusing to open non-HTTPS URL: {url}"
        raise ValueError(msg)


def fetch_objects(url: str) -> list[str]:
    """Fetch the object list from the YCB benchmark index URL."""
    _validate_https_url(url)
    with urlopen(url) as response:  # noqa: S310  # scheme restricted to HTTPS by _validate_https_url
        payload = json.loads(response.read())
    objects = payload["objects"]
    if not isinstance(objects, list):
        msg = "YCB object index did not contain a list of objects"
        raise TypeError(msg)
    return [str(object_id) for object_id in objects]


def download_file(url: str, filename: Path) -> None:
    """Download a file from ``url`` with retries."""
    _validate_https_url(url)
    for attempt in range(MAX_RETRIES):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310  # HTTPS-only config URL
            with urlopen(request, timeout=TIMEOUT_SECONDS) as remote_file:  # noqa: S310  # HTTPS-only config URL
                file_size_header = remote_file.getheader("Content-Length")
                file_size = int(file_size_header) if file_size_header else 0
                logger.info("Downloading: {} ({:.2f} MB)", filename, file_size / 1_000_000.0)

                file_size_dl = 0
                with filename.open("wb") as local_file:
                    while True:
                        buffer = remote_file.read(BLOCK_SIZE)
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
            logger.warning("Error downloading (attempt {}/{}): {}", attempt + 1, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP_SECONDS)
            else:
                raise
        else:
            return


def tgz_url(ycb_object: str, file_type: str) -> str:
    """Return the TGZ file URL for a YCB object and dataset type."""
    if file_type in BERKELEY_RGB_TYPES:
        return f"{BASE_URL}berkeley/{ycb_object}/{ycb_object}_{file_type}.tgz"
    if file_type == "berkeley_processed":
        return f"{BASE_URL}berkeley/{ycb_object}/{ycb_object}_berkeley_meshes.tgz"
    return f"{BASE_URL}google/{ycb_object}_{file_type}.tgz"


def extract_tgz(filename: Path, output_dir: Path) -> None:
    """Extract a TGZ archive into ``output_dir``."""
    with tarfile.open(filename, "r:gz") as archive:
        archive.extractall(path=output_dir, filter="data")

    for _ in range(UNPACK_DELETE_RETRIES):
        try:
            filename.unlink()
            break
        except OSError:
            time.sleep(CLEANUP_SLEEP_SECONDS)


def check_url(url: str) -> bool:
    """Return whether ``url`` responds to a HEAD request."""
    _validate_https_url(url)
    try:
        request = Request(url)  # noqa: S310  # HTTPS-only config URL
        request.get_method = lambda: "HEAD"
        with urlopen(request):  # noqa: S310  # HTTPS-only config URL
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


def _cleanup_failed_object(object_id: str) -> None:
    object_dir = OUTPUT_DIRECTORY / object_id
    if object_dir.exists():
        shutil.rmtree(object_dir, ignore_errors=True)

    for tgz_file in OUTPUT_DIRECTORY.glob(f"{object_id}_*.tgz"):
        for _ in range(CLEANUP_DELETE_RETRIES):
            try:
                tgz_file.unlink()
                break
            except OSError:
                time.sleep(CLEANUP_SLEEP_SECONDS)


def _process_object(object_id: str) -> None:
    object_dir = OUTPUT_DIRECTORY / object_id
    for file_type in FILES_TO_DOWNLOAD:
        if EXTRACT:
            if _is_already_extracted(object_dir, file_type):
                logger.info("Skipping {} {}: already extracted.", object_id, file_type)
                continue
        else:
            filename = OUTPUT_DIRECTORY / f"{object_id}_{file_type}.tgz"
            if filename.is_file():
                logger.info("Skipping {} {}: already downloaded.", object_id, file_type)
                continue

        url = tgz_url(object_id, file_type)
        if not check_url(url):
            continue

        filename = OUTPUT_DIRECTORY / f"{object_id}_{file_type}.tgz"
        download_file(url, filename)
        if EXTRACT:
            extract_tgz(filename, OUTPUT_DIRECTORY)


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/download_ycb_dataset")
def main(_cfg: DictConfig) -> None:
    """Download and extract the configured YCB object archives."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    objects = fetch_objects(OBJECTS_URL)

    for object_id in objects:
        if object_id not in OBJECTS_TO_DOWNLOAD:
            continue

        try:
            _process_object(object_id)
        except (OSError, URLError) as exc:
            logger.error("Failed to process {}: {}. Cleaning up...", object_id, exc)
            _cleanup_failed_object(object_id)


if __name__ == "__main__":
    main()
