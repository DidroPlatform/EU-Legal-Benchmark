from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.runner.response_sources import load_prefilled_responses, load_previous_output_responses


class TestResponseSourceParsing(unittest.TestCase):
    def test_prefilled_jsonl_valid_row_loads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prefilled.jsonl"
            path.write_text(
                '{"example_id":"ex1","candidate_name":"cand_a","response_text":"answer"}\n',
                encoding="utf-8",
            )
            parsed = load_prefilled_responses(str(path))
            self.assertEqual(parsed, {("ex1", "cand_a"): "answer"})

    def test_prefilled_jsonl_invalid_json_reports_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prefilled.jsonl"
            path.write_text("{\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_prefilled_responses(str(path))
            self.assertIn("Invalid JSON in prefilled responses at line 1", str(ctx.exception))

    def test_prefilled_jsonl_rejects_non_object_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prefilled.jsonl"
            path.write_text('"text"\n', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_prefilled_responses(str(path))
            self.assertIn("Invalid prefilled response row at line 1: must be an object.", str(ctx.exception))

    def test_prefilled_jsonl_requires_example_id_and_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prefilled.jsonl"
            path.write_text('{"example_id":"","candidate_name":"cand_a","response_text":"x"}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_prefilled_responses(str(path))
            self.assertIn("`example_id` and `candidate_name` are required", str(ctx.exception))

    def test_prefilled_jsonl_requires_string_response_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prefilled.jsonl"
            path.write_text('{"example_id":"ex1","candidate_name":"cand_a","response_text":1}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_prefilled_responses(str(path))
            self.assertIn("`response_text` must be a string", str(ctx.exception))

    def test_prefilled_jsonl_rejects_duplicate_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prefilled.jsonl"
            path.write_text(
                '{"example_id":"ex1","candidate_name":"cand_a","response_text":"one"}\n'
                '{"example_id":"ex1","candidate_name":"cand_a","response_text":"two"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_prefilled_responses(str(path))
            self.assertIn("Duplicate prefilled response for example_id=ex1, candidate_name=cand_a", str(ctx.exception))

    def test_previous_output_json_list_of_objects_loads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "responses.json"
            path.write_text(
                '[{"example_id":"ex1","candidate_name":"cand_a","response_text":"reused"}]\n',
                encoding="utf-8",
            )
            parsed = load_previous_output_responses(str(path), candidate_names=["cand_a"])
            self.assertEqual(parsed, {("ex1", "cand_a"): "reused"})

    def test_previous_output_simple_dict_loads_for_single_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "responses.json"
            path.write_text('{"ex1":"reused"}\n', encoding="utf-8")
            parsed = load_previous_output_responses(str(path), candidate_names=["cand_a"])
            self.assertEqual(parsed, {("ex1", "cand_a"): "reused"})

    def test_previous_output_simple_dict_is_ambiguous_for_multiple_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "responses.json"
            path.write_text('{"ex1":"reused"}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_previous_output_responses(str(path), candidate_names=["cand_a", "cand_b"])
            self.assertIn("Ambiguous previous output JSON mapping", str(ctx.exception))

    def test_previous_output_rejects_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "responses.txt"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_previous_output_responses(str(path), candidate_names=["cand_a"])
            self.assertIn("Unsupported previous output file extension '.txt'. Expected .jsonl or .json.", str(ctx.exception))

