from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.backfill_run import collect_backfill_targets, create_filtered_datasets


class TestBackfillRun(unittest.TestCase):
    def test_collect_backfill_targets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outputs = Path(td)
            (outputs / "summary.json").write_text(
                json.dumps(
                    {
                        "failed_items": [
                            {"dataset": "d", "example_id": "ex1", "candidate_name": "cand_a"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "judgments.jsonl").write_text(
                json.dumps(
                    {
                        "dataset": "d",
                        "example_id": "ex2",
                        "candidate_name": "cand_b",
                        "parse_error": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (outputs / "responses.jsonl").write_text(
                json.dumps(
                    {
                        "dataset": "d",
                        "example_id": "ex3",
                        "candidate_name": "cand_c",
                        "response_text": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            targets = collect_backfill_targets(
                outputs,
                include_failed_generation=True,
                include_parse_errors=True,
                include_empty_responses=True,
            )
            self.assertEqual(
                targets,
                {
                    ("d", "ex1", "cand_a"),
                    ("d", "ex2", "cand_b"),
                    ("d", "ex3", "cand_c"),
                },
            )

    def test_create_filtered_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_file = root / "source.jsonl"
            dataset_file.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "ex1", "prompt": "p1"}),
                        json.dumps({"id": "ex2", "prompt": "p2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            run_config = {
                "datasets": [
                    {
                        "name": "d",
                        "path": str(dataset_file),
                        "provenance": "canonical:test",
                        "judge_mode": "reference",
                        "enabled": True,
                    }
                ]
            }
            out = create_filtered_datasets(
                run_config=run_config,
                dataset_to_example_ids={"d": {"ex2"}},
                output_dir=root / "filtered",
            )
            self.assertEqual(len(out), 1)
            filtered_path = Path(out[0]["path"])
            rows = [json.loads(line) for line in filtered_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["id"] for row in rows], ["ex2"])

