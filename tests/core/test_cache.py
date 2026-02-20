from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.cache import DiskCache


class TestDiskCache(unittest.TestCase):
    def test_delete_removes_existing_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = DiskCache(td, enabled=True)
            key = cache.make_key({"a": 1})
            cache.set(key, {"text": "ok"})
            self.assertIsNotNone(cache.get(key))
            cache.delete(key)
            self.assertIsNone(cache.get(key))

    def test_delete_missing_key_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = DiskCache(td, enabled=True)
            key = "does-not-exist"
            cache.delete(key)
            self.assertFalse((Path(td) / f"{key}.json").exists())

