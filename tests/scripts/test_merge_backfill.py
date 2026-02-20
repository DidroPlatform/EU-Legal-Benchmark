from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_backfill import merge_run_outputs


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class TestMergeBackfill(unittest.TestCase):
    def test_merge_overlays_backfill_rows_and_recomputes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "base_run"
            backfill = root / "backfill_run"
            out = root / "merged_run"
            (base / "outputs").mkdir(parents=True)
            (backfill / "outputs").mkdir(parents=True)

            examples = [{"example_id": "ex1", "dataset": "d"}, {"example_id": "ex2", "dataset": "d"}]
            base_responses = [
                {"dataset": "d", "example_id": "ex1", "candidate_name": "cand", "response_text": "old", "run_id": "base"},
            ]
            base_judgments = [
                {
                    "dataset": "d",
                    "example_id": "ex1",
                    "candidate_name": "cand",
                    "score": 0.0,
                    "pass": False,
                    "judge_name": "j",
                    "judge_provider": "p",
                    "judge_model": "m",
                    "judge_settings": {},
                    "request_id": "r1",
                    "cache_key": None,
                    "cache_hit": False,
                    "rationale": "bad",
                    "criteria": {},
                    "parse_error": True,
                    "raw_judge": {},
                    "run_id": "base",
                    "run_started_at_utc": "t0",
                    "provenance": "canonical:test",
                    "judge_mode": "reference",
                }
            ]
            patch_responses = [
                {"dataset": "d", "example_id": "ex1", "candidate_name": "cand", "response_text": "new", "run_id": "patch"},
                {"dataset": "d", "example_id": "ex2", "candidate_name": "cand", "response_text": "new2", "run_id": "patch"},
            ]
            patch_judgments = [
                {
                    "dataset": "d",
                    "example_id": "ex1",
                    "candidate_name": "cand",
                    "score": 1.0,
                    "pass": True,
                    "judge_name": "j",
                    "judge_provider": "p",
                    "judge_model": "m",
                    "judge_settings": {},
                    "request_id": "r2",
                    "cache_key": None,
                    "cache_hit": False,
                    "rationale": "good",
                    "criteria": {},
                    "parse_error": False,
                    "raw_judge": {},
                    "run_id": "patch",
                    "run_started_at_utc": "t1",
                    "provenance": "canonical:test",
                    "judge_mode": "reference",
                },
                {
                    "dataset": "d",
                    "example_id": "ex2",
                    "candidate_name": "cand",
                    "score": 1.0,
                    "pass": True,
                    "judge_name": "j",
                    "judge_provider": "p",
                    "judge_model": "m",
                    "judge_settings": {},
                    "request_id": "r3",
                    "cache_key": None,
                    "cache_hit": False,
                    "rationale": "good2",
                    "criteria": {},
                    "parse_error": False,
                    "raw_judge": {},
                    "run_id": "patch",
                    "run_started_at_utc": "t1",
                    "provenance": "canonical:test",
                    "judge_mode": "reference",
                },
            ]

            _write_jsonl(base / "outputs" / "examples.jsonl", examples)
            _write_jsonl(base / "outputs" / "responses.jsonl", base_responses)
            _write_jsonl(base / "outputs" / "judgments.jsonl", base_judgments)
            _write_jsonl(base / "outputs" / "trace.jsonl", [])
            _write_json(
                base / "outputs" / "summary.json",
                {
                    "run_id": "base_run",
                    "run_started_at_utc": "2026-01-01T00:00:00+00:00",
                    "selected_examples": 2,
                    "datasets": [],
                    "judge": {},
                    "judges": [],
                },
            )
            _write_json(base / "outputs" / "run_config.json", {"candidates": [{"name": "cand"}]})

            _write_jsonl(backfill / "outputs" / "responses.jsonl", patch_responses)
            _write_jsonl(backfill / "outputs" / "judgments.jsonl", patch_judgments)
            _write_jsonl(backfill / "outputs" / "trace.jsonl", [])

            report = merge_run_outputs(base, backfill, out)
            self.assertEqual(report["replaced_responses"], 1)
            self.assertEqual(report["replaced_judgments"], 1)
            self.assertEqual(report["missing_responses_after_merge"], 0)
            self.assertEqual(report["missing_judgments_after_merge"], 0)

            merged_summary = json.loads((out / "outputs" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(merged_summary["num_responses"], 2)
            self.assertEqual(merged_summary["num_judgments"], 2)
            self.assertEqual(merged_summary["run_status"], "completed")
            self.assertEqual(merged_summary["num_failures"], 0)

            merged_responses = [
                json.loads(line)
                for line in (out / "outputs" / "responses.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_example = {row["example_id"]: row for row in merged_responses}
            self.assertEqual(by_example["ex1"]["response_text"], "new")
            self.assertEqual(by_example["ex2"]["response_text"], "new2")

    def test_merge_marks_degraded_when_expected_pairs_are_still_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "base_run"
            backfill = root / "backfill_run"
            out = root / "merged_run"
            (base / "outputs").mkdir(parents=True)
            (backfill / "outputs").mkdir(parents=True)

            examples = [{"example_id": "ex1", "dataset": "d"}, {"example_id": "ex2", "dataset": "d"}]
            base_responses = [
                {"dataset": "d", "example_id": "ex1", "candidate_name": "cand", "response_text": "base"},
            ]
            base_judgments = [
                {
                    "dataset": "d",
                    "example_id": "ex1",
                    "candidate_name": "cand",
                    "score": 1.0,
                    "pass": True,
                    "judge_name": "j",
                    "judge_provider": "p",
                    "judge_model": "m",
                    "judge_settings": {},
                    "request_id": "r1",
                    "cache_key": None,
                    "cache_hit": False,
                    "rationale": "ok",
                    "criteria": {},
                    "parse_error": False,
                    "raw_judge": {},
                    "run_id": "base",
                    "run_started_at_utc": "t0",
                    "provenance": "canonical:test",
                    "judge_mode": "reference",
                }
            ]

            _write_jsonl(base / "outputs" / "examples.jsonl", examples)
            _write_jsonl(base / "outputs" / "responses.jsonl", base_responses)
            _write_jsonl(base / "outputs" / "judgments.jsonl", base_judgments)
            _write_jsonl(base / "outputs" / "trace.jsonl", [])
            _write_json(
                base / "outputs" / "summary.json",
                {
                    "run_id": "base_run",
                    "run_started_at_utc": "2026-01-01T00:00:00+00:00",
                    "selected_examples": 2,
                    "datasets": [],
                    "judge": {},
                    "judges": [],
                },
            )
            _write_json(base / "outputs" / "run_config.json", {"candidates": [{"name": "cand"}]})

            _write_jsonl(backfill / "outputs" / "responses.jsonl", [])
            _write_jsonl(backfill / "outputs" / "judgments.jsonl", [])
            _write_jsonl(backfill / "outputs" / "trace.jsonl", [])

            report = merge_run_outputs(base, backfill, out)
            self.assertEqual(report["missing_responses_after_merge"], 1)
            self.assertEqual(report["missing_judgments_after_merge"], 1)

            merged_summary = json.loads((out / "outputs" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(merged_summary["run_status"], "degraded")
            self.assertEqual(merged_summary["num_failures"], 2)
            failed_items = merged_summary["failed_items"]
            self.assertEqual(len(failed_items), 2)
            self.assertTrue(any(item["example_id"] == "ex2" and item["stage"] == "response_missing_after_merge" for item in failed_items))
            self.assertTrue(any(item["example_id"] == "ex2" and item["stage"] == "judge_missing_after_merge" for item in failed_items))

