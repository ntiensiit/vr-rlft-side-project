"""Download YCB object assets used by simulation and data prep."""

from __future__ import annotations

import glob
import json
import shutil
import tarfile
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DownloadSettings:
    base_url: str
    user_agent: str
    block_size: int
    max_retries: int
    timeout_seconds: int
    retry_sleep_seconds: float
    unpack_delete_retries: int
    cleanup_delete_retries: int
    cleanup_sleep_seconds: float
    berkeley_rgb_types: tuple[str, ...]


def fetch_objects(url: str) -> list[str]:
    """Fetch the object list from the YCB benchmark index URL."""
    with urlopen(url) as response:
        payload = json.loads(response.read())
    objects = payload["objects"]
    if not isinstance(objects, list):
        msg = "YCB object index did not contain a list of objects"
        raise TypeError(msg)
    return [str(object_id) for object_id in objects]


def download_file(url: str, filename: Path, settings: DownloadSettings) -> None:
    """Download a file from ``url`` with retries."""
    for attempt in range(settings.max_retries):
        try:
            request = Request(url, headers={"User-Agent": settings.user_agent})
            with urlopen(request, timeout=settings.timeout_seconds) as remote_file:
                file_size_header = remote_file.getheader("Content-Length")
                file_size = int(file_size_header) if file_size_header else 0
                logger.info("Downloading: {} ({:.2f} MB)", filename, file_size / 1_000_000.0)

                file_size_dl = 0
                with filename.open("wb") as local_file:
                    while True:
                        buffer = remote_file.read(settings.block_size)
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
            logger.warning("Error downloading (attempt {}/{}): {}", attempt + 1, settings.max_retries, exc)
            if attempt < settings.max_retries - 1:
                time.sleep(settings.retry_sleep_seconds)
            else:
                raise
        else:
            return


def tgz_url(ycb_object: str, file_type: str, settings: DownloadSettings) -> str:
    """Return the TGZ file URL for a YCB object and dataset type."""
    if file_type in settings.berkeley_rgb_types:
        return f"{settings.base_url}berkeley/{ycb_object}/{ycb_object}_{file_type}.tgz"
    if file_type == "berkeley_processed":
        return f"{settings.base_url}berkeley/{ycb_object}/{ycb_object}_berkeley_meshes.tgz"
    return f"{settings.base_url}google/{ycb_object}_{file_type}.tgz"


def extract_tgz(filename: Path, output_dir: Path, settings: DownloadSettings) -> None:
    """Extract a TGZ archive into ``output_dir``."""
    with tarfile.open(filename, "r:gz") as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(path=output_dir, filter="data")
        else:
            archive.extractall(path=output_dir)

    for _ in range(settings.unpack_delete_retries):
        try:
            filename.unlink()
            break
        except OSError:
            time.sleep(settings.cleanup_sleep_seconds)


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


def _cleanup_failed_object(output_directory: Path, object_id: str, settings: DownloadSettings) -> None:
    object_dir = output_directory / object_id
    if object_dir.exists():
        shutil.rmtree(object_dir, ignore_errors=True)

    tgz_pattern = output_directory / f"{object_id}_*.tgz"
    for file_name in glob.glob(str(tgz_pattern)):
        for _ in range(settings.cleanup_delete_retries):
            try:
                Path(file_name).unlink()
                break
            except OSError:
                time.sleep(settings.cleanup_sleep_seconds)


def _process_object(
    object_id: str,
    output_directory: Path,
    files_to_download: list[str],
    extract: bool,
    settings: DownloadSettings,
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

        url = tgz_url(object_id, file_type, settings)
        if not check_url(url):
            continue

        filename = output_directory / f"{object_id}_{file_type}.tgz"
        download_file(url, filename, settings)
        if extract:
            extract_tgz(filename, output_directory, settings)


def _download_settings(cfg: DictConfig) -> DownloadSettings:
    berkeley_rgb_types = config_value(cfg, "download", "berkeley_rgb_types", value_type=list[str])
    return DownloadSettings(
        base_url=str(config_value(cfg, "download", "base_url", value_type=object, required=True)),
        user_agent=str(config_value(cfg, "download", "user_agent", value_type=object, required=True)),
        block_size=config_value(cfg, "download", "block_size", value_type=int, required=True),
        max_retries=config_value(cfg, "download", "max_retries", value_type=int, required=True),
        timeout_seconds=config_value(cfg, "download", "timeout_seconds", value_type=int, required=True),
        retry_sleep_seconds=config_value(cfg, "download", "retry_sleep_seconds", value_type=float, required=True),
        unpack_delete_retries=config_value(cfg, "download", "unpack_delete_retries", value_type=int, required=True),
        cleanup_delete_retries=config_value(cfg, "download", "cleanup_delete_retries", value_type=int, required=True),
        cleanup_sleep_seconds=config_value(cfg, "download", "cleanup_sleep_seconds", value_type=float, required=True),
        berkeley_rgb_types=tuple(berkeley_rgb_types),
    )


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/download_ycb_dataset")
def main(cfg: DictConfig) -> None:
    settings = _download_settings(cfg)
    output_directory = config_value(
        cfg, "output_directory", "download", "output_directory", value_type=Path, script_or=True
    )
    objects_to_download = config_value(cfg, "objects_to_download", "download", "objects", value_type=list[str], script_or=True)
    files_to_download = config_value(cfg, "files_to_download", "download", "files", value_type=list[str], script_or=True)
    extract = config_value(cfg, "extract", "download", "extract", value_type=bool, script_or=True)
    objects_url = str(config_value(cfg, "objects_url", "download", "objects_url", value_type=object, script_or=True))

    output_directory.mkdir(parents=True, exist_ok=True)
    objects = fetch_objects(objects_url)

    for object_id in objects:
        if object_id not in objects_to_download:
            continue

        try:
            _process_object(object_id, output_directory, files_to_download, extract, settings)
        except (OSError, URLError) as exc:
            logger.error("Failed to process {}: {}. Cleaning up...", object_id, exc)
            _cleanup_failed_object(output_directory, object_id, settings)


if __name__ == "__main__":
    main()
