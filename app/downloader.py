from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https")


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    return name if name else "video.mp4"


def download_video(source: str, dest_dir: str | None = None) -> Path:
    """Download a video URL or validate a local path; return the local Path."""
    if not _is_url(source):
        local = Path(source)
        if not local.exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        if not local.is_file():
            raise ValueError(f"Path is not a file: {source}")
        return local.resolve()

    managed = dest_dir is None
    work_dir = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="vcap_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    dest = work_dir / _filename_from_url(source)

    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task(f"Downloading {Path(source).name}", total=None)

            with httpx.stream("GET", source, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0)) or None
                progress.update(task, total=total)

                with dest.open("wb") as fh:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        fh.write(chunk)
                        progress.advance(task, len(chunk))
    except Exception:
        if managed:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise

    return dest


def cleanup_download(path: Path) -> None:
    """Remove the temp directory created by download_video, if it was a temp dir."""
    if path.parent.name.startswith("vcap_"):
        shutil.rmtree(path.parent, ignore_errors=True)
