from __future__ import annotations

import builtins
import unittest
from unittest import mock

from src.runtime.bootstrap import load_dotenv_if_available


class TestRuntimeBootstrap(unittest.TestCase):
    def test_load_dotenv_if_available_is_noop_when_dotenv_missing(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "dotenv":
                raise ImportError("dotenv not installed")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            load_dotenv_if_available()

