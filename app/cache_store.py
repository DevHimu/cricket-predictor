"""Disk cache for computed match analysis.

Completed matches never change, so their analysis is computed once and served
from a folder afterwards. Live matches are never cached.

Folder: <project>/data/analysis/<match_id>.json  (override with ANALYSIS_CACHE_DIR)

Note: on Render's free tier the container filesystem is ephemeral, so this cache
is cleared on restart/redeploy. That is fine - it is a cache, not a database.
It removes repeated scorer round-trips and recomputation while the service is
up. For durable storage, point ANALYSIS_CACHE_DIR at a mounted disk.
"""
import json
import os
import re
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(os.path.dirname(HERE), "data", "analysis")
CACHE_DIR = os.environ.get("ANALYSIS_CACHE_DIR", DEFAULT_DIR)

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _path(match_id: str) -> str:
    return os.path.join(CACHE_DIR, _SAFE.sub("_", str(match_id)) + ".json")


def ensure_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def get(match_id: str):
    try:
        with open(_path(match_id), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def put(match_id: str, data) -> bool:
    """Atomic write so a crash mid-write can't leave a corrupt cache file."""
    try:
        ensure_dir()
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, _path(match_id))
        return True
    except OSError:
        return False


def clear(match_id: str) -> bool:
    try:
        os.remove(_path(match_id))
        return True
    except OSError:
        return False


def info() -> dict:
    try:
        files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
    except OSError:
        files = []
    return {"dir": CACHE_DIR, "cached_matches": len(files), "files": files[:50]}
