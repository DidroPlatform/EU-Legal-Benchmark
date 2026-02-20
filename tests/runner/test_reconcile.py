from __future__ import annotations

import unittest

from src.runner.reconcile import overlay_rows


class TestReconcile(unittest.TestCase):
    def test_overlay_rows_replaces_and_sorts_by_composite_key(self) -> None:
        base_rows = [
            {"dataset": "d", "example_id": "ex2", "candidate_name": "c", "value": "base-ex2"},
            {"dataset": "d", "example_id": "ex1", "candidate_name": "c", "value": "base-ex1"},
        ]
        patch_rows = [
            {"dataset": "d", "example_id": "ex2", "candidate_name": "c", "value": "patch-ex2"},
            {"dataset": "d", "example_id": "ex3", "candidate_name": "c", "value": "patch-ex3"},
        ]

        merged, replaced = overlay_rows(base_rows, patch_rows)

        self.assertEqual(replaced, 1)
        self.assertEqual([row["example_id"] for row in merged], ["ex1", "ex2", "ex3"])
        self.assertEqual(merged[1]["value"], "patch-ex2")

