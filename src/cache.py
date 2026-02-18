from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _log_warning(message: str) -> None:
    """Log a cache warning to stderr."""
    print(f"[cache warning] {message}", file=sys.stderr)


class DiskCache:
    """Simple file-per-key cache for deterministic-ish benchmark reruns."""

    def __init__(self, root: str, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            _log_warning(f"Corrupted cache entry {path.name}, discarding: {exc}")
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path_for_key(key)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        tmp_path.replace(path)
