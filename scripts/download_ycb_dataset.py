"""Download YCB object assets used by simulation and data prep."""

from __future__ import annotations

import asyncio
import json
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError

import aiohttp
import hydra
from loguru import logger

from grasping_ai.config import FLATTENED_YAML_CONFIG, SCRIPTS_CONFIG_PATH, FlattenedYAMLConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig

OUTPUT_DIRECTORY = Path(str(FLATTENED_YAML_CONFIG.get("script.output_directory")))
OBJECTS_TO_DOWNLOAD = [str(object_id) for object_id in FLATTENED_YAML_CONFIG.get("script.objects_to_download", [])]
FILES_TO_DOWNLOAD = [str(file_type) for file_type in FLATTENED_YAML_CONFIG.get("script.files_to_download", [])]
EXTRACT = bool(FLATTENED_YAML_CONFIG.get("script.extract", True))
OBJECTS_URL = str(FLATTENED_YAML_CONFIG.get("script.objects_url"))
BASE_URL = str(FLATTENED_YAML_CONFIG.get("script.base_url"))
USER_AGENT = str(FLATTENED_YAML_CONFIG.get("script.user_agent"))
BLOCK_SIZE = int(FLATTENED_YAML_CONFIG.get("script.block_size", 65536))
MAX_WORKERS = int(FLATTENED_YAML_CONFIG.get("script.max_workers", 8))
MAX_THREADS = int(FLATTENED_YAML_CONFIG.get("script.max_threads", 4))
MAX_RETRIES = int(FLATTENED_YAML_CONFIG.get("script.max_retries", 5))
TIMEOUT_SECONDS = int(FLATTENED_YAML_CONFIG.get("script.timeout_seconds", 30))
RETRY_SLEEP_SECONDS = float(FLATTENED_YAML_CONFIG.get("script.retry_sleep_seconds", 2))
UNPACK_DELETE_RETRIES = int(FLATTENED_YAML_CONFIG.get("script.unpack_delete_retries", 10))
CLEANUP_DELETE_RETRIES = int(FLATTENED_YAML_CONFIG.get("script.cleanup_delete_retries", 5))
CLEANUP_SLEEP_SECONDS = float(FLATTENED_YAML_CONFIG.get("script.cleanup_sleep_seconds", 1))
BERKELEY_RGB_TYPES = tuple(FLATTENED_YAML_CONFIG.get("script.berkeley_rgb_types", []))


def _validate_https_url(url: str) -> None:
    """Raise ``ValueError`` unless ``url`` uses the HTTPS scheme."""
    if not url.startswith("https://"):
        msg = f"Refusing to open non-HTTPS URL: {url}"
        raise ValueError(msg)


async def fetch_objects(url: str) -> list[str]:
    """Fetch the object list from the YCB benchmark index URL."""
    _validate_https_url(url)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session, session.get(
        url, headers={"User-Agent": USER_AGENT},
    ) as response:
        response.raise_for_status()
        payload = json.loads(await response.text())
    objects = payload["objects"]
    if not isinstance(objects, list):
        msg = "YCB object index did not contain a list of objects"
        raise TypeError(msg)
    return [str(object_id) for object_id in objects]


async def _download_file(url: str, filename: Path, session: aiohttp.ClientSession) -> None:
    """Download a file from ``url`` to ``filename`` with retries."""
    _validate_https_url(url)
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, headers={"User-Agent": USER_AGENT}) as remote_file:
                remote_file.raise_for_status()
                file_size_header = remote_file.headers.get("Content-Length")
                file_size = int(file_size_header) if file_size_header else 0
                logger.info("Downloading: {} ({:.2f} MB)", filename, file_size / 1_000_000.0)

                file_size_dl = 0
                with filename.open("wb") as local_file:
                    async for buffer in remote_file.content.iter_chunked(BLOCK_SIZE):
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
                        logger.info("{}", status)
        except (OSError, URLError, aiohttp.ClientError) as exc:
            logger.warning("Error downloading (attempt {}/{}): {}", attempt + 1, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_SLEEP_SECONDS)
            else:
                _remove_tgz(filename)
                raise
        else:
            return


def tgz_url(ycb_object: str, file_type: str, base_url: str, berkeley_rgb_types: list[str]) -> str:
    """Return the TGZ file URL for a YCB object and dataset type."""
    if file_type in berkeley_rgb_types:
        return f"{base_url}berkeley/{ycb_object}/{ycb_object}_{file_type}.tgz"
    if file_type == "berkeley_processed":
        return f"{base_url}berkeley/{ycb_object}/{ycb_object}_berkeley_meshes.tgz"
    return f"{base_url}google/{ycb_object}_{file_type}.tgz"


def extract_tgz(filename: Path, output_dir: Path) -> None:
    """Extract a TGZ archive into ``output_dir``.

    On any failure the archive is removed so a corrupt download is not kept.
    """
    try:
        with tarfile.open(filename, "r:gz") as archive:
            archive.extractall(path=output_dir, filter="data")
    except BaseException:
        _remove_tgz(filename)
        raise

    for _ in range(UNPACK_DELETE_RETRIES):
        try:
            filename.unlink()
            break
        except OSError:
            time.sleep(CLEANUP_SLEEP_SECONDS)


def _remove_tgz(filename: Path) -> None:
    """Best-effort removal of a (partial or corrupt) TGZ file."""
    for _ in range(CLEANUP_DELETE_RETRIES):
        try:
            filename.unlink()
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(CLEANUP_SLEEP_SECONDS)
        else:
            return


async def _check_url(url: str, session: aiohttp.ClientSession) -> bool:
    """Return whether ``url`` responds to a HEAD request."""
    _validate_https_url(url)
    try:
        async with session.head(url) as response:
            return response.status < 400  # noqa: PLR2004  # any 4xx/5xx is treated as unavailable
    except (OSError, URLError, aiohttp.ClientError):
        return False


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

    for tgz_file in output_directory.glob(f"{object_id}_*.tgz"):
        for _ in range(CLEANUP_DELETE_RETRIES):
            try:
                tgz_file.unlink()
                break
            except OSError:
                time.sleep(CLEANUP_SLEEP_SECONDS)


async def _process_object(  # noqa: PLR0913, PLR0917  # plain data for one download task
    object_id: str,
    output_directory: Path,
    files_to_download: list[str],
    berkeley_rgb_types: list[str],
    base_url: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> None:
    """Download and extract the configured archives for ``object_id``."""
    object_dir = output_directory / object_id
    for file_type in files_to_download:
        if EXTRACT:
            if _is_already_extracted(object_dir, file_type):
                logger.info("Skipping {} {}: already extracted.", object_id, file_type)
                continue
        else:
            filename = output_directory / f"{object_id}_{file_type}.tgz"
            if filename.is_file():
                logger.info("Skipping {} {}: already downloaded.", object_id, file_type)
                continue

        url = tgz_url(object_id, file_type, base_url, berkeley_rgb_types)
        if not await _check_url(url, session):
            continue

        filename = output_directory / f"{object_id}_{file_type}.tgz"
        async with semaphore:
            await _download_file(url, filename, session)
        if EXTRACT:
            await asyncio.to_thread(extract_tgz, filename, output_directory)


@hydra.main(version_base=None, config_path=SCRIPTS_CONFIG_PATH, config_name="scripts/download_ycb_dataset")
def main(cfg: DictConfig) -> None:
    """Download and extract the configured YCB object archives."""
    yaml_config = FlattenedYAMLConfig(cfg)
    output_directory = yaml_config.value(
        "output_directory",
        "download",
        "output_directory",
        value_type=Path,
        script_or=True,
        default=OUTPUT_DIRECTORY,
    )
    objects_to_download = yaml_config.value(
        "objects_to_download",
        "download",
        "objects",
        value_type=list[str],
        script_or=True,
        default=OBJECTS_TO_DOWNLOAD,
    )
    objects_url = str(
        yaml_config.value(
            "objects_url", "download", "objects_url", value_type=object, script_or=True, default=OBJECTS_URL,
        ),
    )
    files_to_download = yaml_config.value(
        "files_to_download",
        "download",
        "files",
        value_type=list[str],
        script_or=True,
        default=FILES_TO_DOWNLOAD,
    )
    berkeley_rgb_types = yaml_config.value(
        "berkeley_rgb_types",
        "download",
        "berkeley_rgb_types",
        value_type=list[str],
        script_or=True,
        default=BERKELEY_RGB_TYPES,
    )
    base_url = str(
        yaml_config.value(
            "base_url", "download", "base_url", value_type=object, script_or=True, default=BASE_URL,
        ),
    )
    max_workers = yaml_config.value(
        "max_workers", "download", "max_workers", value_type=int, script_or=True, default=MAX_WORKERS,
    )
    max_threads = yaml_config.value(
        "max_threads", "download", "max_threads", value_type=int, script_or=True, default=MAX_THREADS,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    _run_downloads(
        output_directory,
        objects_to_download,
        objects_url,
        files_to_download,
        berkeley_rgb_types,
        base_url,
        max_workers,
        max_threads,
    )


def _run_objects_in_thread(  # noqa: PLR0913, PLR0917  # plain data for one thread's download task
    object_ids: list[str],
    output_directory: Path,
    files_to_download: list[str],
    berkeley_rgb_types: list[str],
    base_url: str,
    max_workers: int,
) -> list[tuple[str, BaseException | None]]:
    """Run an asyncio event loop for ``object_ids`` inside one worker thread."""

    async def _amain() -> list[BaseException | None]:
        semaphore = asyncio.Semaphore(max_workers)
        timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await asyncio.gather(
                *(
                    _process_object(
                        object_id,
                        output_directory,
                        files_to_download,
                        berkeley_rgb_types,
                        base_url,
                        session,
                        semaphore,
                    )
                    for object_id in object_ids
                ),
                return_exceptions=True,
            )

    results = asyncio.run(_amain())
    return list(zip(object_ids, results, strict=True))


def _run_downloads(  # noqa: PLR0913, PLR0917  # task carries the resolved download configuration
    output_directory: Path,
    objects_to_download: list[str],
    objects_url: str,
    files_to_download: list[str],
    berkeley_rgb_types: list[str],
    base_url: str,
    max_workers: int,
    max_threads: int,
) -> None:
    """Download and extract all configured objects, combining threads with asyncio."""
    objects = asyncio.run(fetch_objects(objects_url))
    targets = [object_id for object_id in objects if object_id in objects_to_download]
    chunks: list[list[str]] = [targets[i::max_threads] for i in range(max_threads)]
    chunks = [chunk for chunk in chunks if chunk]

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(
                _run_objects_in_thread,
                chunk,
                output_directory,
                files_to_download,
                berkeley_rgb_types,
                base_url,
                max_workers,
            ): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            chunk = futures[future]
            results = future.result()
            for object_id, result in zip(chunk, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error("Failed to process {}: {}. Cleaning up...", object_id, result)
                    _cleanup_failed_object(output_directory, object_id)


if __name__ == "__main__":
    main()
