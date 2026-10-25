"""Find nuScenes on disk, or download the parts of v1.0-mini this project needs.

The full mini archive is ~4 GB, mostly camera images and LiDAR sweeps. The archive
is streamed and only the requested members are written to disk: the JSON tables
(~40 MB, enough for the object-centric representation and the ego baselines) and,
optionally, the keyframe LiDAR point clouds (~600 MB, used for plots). Nothing else
is stored, so no 4 GB file is left behind.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

MINI_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
ENV_VAR = "NUSCENES_ROOT"
COMMON_LOCATIONS = (
    "data/nuscenes",
    "data/sets/nuscenes",
    "/content/data/sets/nuscenes",
    "/kaggle/working/data/sets/nuscenes",
    "~/data/sets/nuscenes",
)


def has_tables(root: str | Path, version: str = "v1.0-mini") -> bool:
    return (Path(root).expanduser() / version / "sample.json").exists()


def find_nuscenes(candidates=(), version: str = "v1.0-mini") -> Path | None:
    """First folder holding ``<version>/sample.json``: the given candidates, then
    ``$NUSCENES_ROOT``, then common Colab / Kaggle / local locations."""
    roots = [c for c in candidates if c] + [os.environ.get(ENV_VAR, "")] + list(COMMON_LOCATIONS)
    for root in roots:
        if root and has_tables(root, version):
            return Path(root).expanduser()
    return None


def _wanted(name: str, version: str, with_lidar: bool) -> str | None:
    """Path relative to the dataset root if the archive member is needed, else None."""
    parts = Path(name).parts
    for i, part in enumerate(parts):
        if part == version and name.endswith(".json"):
            return str(Path(*parts[i:]))
        if with_lidar and part == "samples" and parts[i + 1 : i + 2] == ("LIDAR_TOP",):
            return str(Path(*parts[i:]))
    return None


def extract_needed(
    fileobj, root: str | Path, version: str = "v1.0-mini", with_lidar: bool = False
) -> int:
    """Stream a .tgz and write only the JSON tables (and keyframe LiDAR) under ``root``.

    Returns the number of files written. Works on a non-seekable stream.
    """
    root = Path(root)
    written = 0
    with tarfile.open(fileobj=fileobj, mode="r|gz") as archive:
        for member in archive:
            rel = _wanted(member.name, version, with_lidar) if member.isfile() else None
            if rel is None:
                continue
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1
    return written


def download_mini(root: str | Path, with_lidar: bool = False, url: str = MINI_URL) -> Path:
    """Download nuScenes v1.0-mini into ``root`` (tables, optionally keyframe LiDAR).

    By downloading you accept the nuScenes terms of use (CC BY-NC-SA 4.0).
    """
    root = Path(root).expanduser()
    print(f"Streaming {url} (~4 GB) and keeping only what is needed ...")
    with urllib.request.urlopen(url) as response:
        written = extract_needed(response, root, with_lidar=with_lidar)
    if not has_tables(root):
        raise RuntimeError(f"download finished but no tables were found in {root}")
    print(f"Wrote {written} files to {root}")
    return root


def find_or_download(
    preferred: str | Path | None = None,
    download_to: str | Path = "data/nuscenes",
    version: str = "v1.0-mini",
    with_lidar: bool = False,
) -> Path:
    """Return a nuScenes root, downloading mini into ``download_to`` if none is found."""
    found = find_nuscenes([str(preferred)] if preferred else [], version)
    if found is not None:
        return found
    if version != "v1.0-mini":
        raise FileNotFoundError(f"nuScenes {version} not found; set ${ENV_VAR} or --data-root")
    return download_mini(download_to, with_lidar=with_lidar)
