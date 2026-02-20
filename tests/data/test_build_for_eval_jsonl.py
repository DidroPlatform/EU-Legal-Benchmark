from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.build_for_eval import _rows_from_source


class TestBuildForEvalJsonl(unittest.TestCase):
    def test_malformed_json_line_is_skipped_with_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(td) / "includebase_europe_law.jsonl"
            source_path.write_text(
                "\n".join(
                    [
                        '{"question":"Q1","option_a":"A1","option_b":"B1","option_c":"C1","option_d":"D1","answer":2}',
                        '{"question":"broken"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows, report = _rows_from_source(source_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(report.rows_read, 2)
            self.assertEqual(report.rows_emitted, 1)
            self.assertEqual(report.rows_invalid, 1)
            self.assertEqual(report.errors[0]["line"], 2)
            self.assertIn("JSON parse error", report.errors[0]["error"])

    def test_includebase_gold_mapping_uses_original_option_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(td) / "includebase_europe_law.jsonl"
            source_path.write_text(
                '{"question":"Q1","option_a":"","option_b":"B1","option_c":"C1","option_d":"D1","answer":1}\n',
                encoding="utf-8",
            )

            rows, report = _rows_from_source(source_path)
            self.assertEqual(report.rows_invalid, 0)
            self.assertEqual(report.rows_emitted, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["choices"][0]["id"], "B")
            self.assertEqual(rows[0]["correct_choice_ids"], ["B"])

    def test_lar_echr_row_maps_to_canonical_mcq(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(td) / "lar_echr_tough_17.jsonl"
            source_path.write_text(
                (
                    '{"record_id":"001-1","case_id":"001-1","case_no":"123/45","facts":"Facts text","context":"Arguments text",'
                    '"a":"Choice A","b":"Choice B","c":"Choice C","d":"Choice D","label":"C","source_split":"test","source_dataset":"AUEB-NLP/lar-echr"}\n'
                ),
                encoding="utf-8",
            )

            rows, report = _rows_from_source(source_path)
            self.assertEqual(report.rows_invalid, 0)
            self.assertEqual(report.rows_emitted, 1)
            self.assertEqual(len(rows), 1)

            row = rows[0]
            self.assertEqual(row["dataset"], "lar_echr")
            self.assertEqual(row["task_type"], "mcq")
            self.assertEqual([c["id"] for c in row["choices"]], ["A", "B", "C", "D"])
            self.assertEqual(row["correct_choice_ids"], ["C"])
            self.assertEqual(row["metadata"]["policy_id"], "lar_echr_mcq_v1")

    def test_lar_echr_invalid_label_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(td) / "lar_echr_tough_17.jsonl"
            source_path.write_text(
                (
                    '{"record_id":"001-1","case_id":"001-1","facts":"Facts text","context":"Arguments text",'
                    '"a":"Choice A","b":"Choice B","label":"D"}\n'
                ),
                encoding="utf-8",
            )

            rows, report = _rows_from_source(source_path)
            self.assertEqual(len(rows), 0)
            self.assertEqual(report.rows_emitted, 0)
            self.assertEqual(report.rows_invalid, 1)
            self.assertIn("label", report.errors[0]["error"].lower())

    def test_prbench_row_reconstructs_messages_and_reference_prepends(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(td) / "prbench_legal_hard_europe.jsonl"
            source_path.write_text(
                (
                    '{"task":"t1","field":"Legal","topic":"T","expert":"Non-Expert","rubric":[{"id":"c1","title":"Crit"}],'
                    '"prompt_0":"First question","response_0":"First answer","reference_texts_0":["Doc A","Doc B"],'
                    '"prompt_1":"Follow-up","response_1":null,"reference_texts_1":[]}'
                    "\n"
                ),
                encoding="utf-8",
            )

            rows, report = _rows_from_source(source_path)
            self.assertEqual(report.rows_invalid, 0)
            self.assertEqual(report.rows_emitted, 1)
            row = rows[0]
            self.assertIn("messages", row)
            self.assertEqual(len(row["messages"]), 3)
            self.assertEqual(row["messages"][0]["role"], "user")
            self.assertIn("Reference Text 0:\nDoc A", row["messages"][0]["content"])
            self.assertIn("First question", row["messages"][0]["content"])
            self.assertEqual(row["messages"][1]["role"], "assistant")
            self.assertEqual(row["messages"][1]["content"], "First answer")
            self.assertEqual(row["prompt"], "Follow-up")

    def test_prbench_response_without_prompt_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(td) / "prbench_legal_hard_europe.jsonl"
            source_path.write_text(
                (
                    '{"task":"t2","field":"Legal","topic":"T","expert":"Non-Expert","rubric":[{"id":"c1","title":"Crit"}],'
                    '"prompt_0":null,"response_0":"orphan","reference_texts_0":[],"prompt_1":"follow up"}'
                    "\n"
                ),
                encoding="utf-8",
            )

            rows, report = _rows_from_source(source_path)
            self.assertEqual(report.rows_invalid, 0)
            self.assertEqual(report.rows_emitted, 1)
            self.assertEqual(rows[0]["messages"][0]["role"], "assistant")
            self.assertEqual(rows[0]["messages"][0]["content"], "orphan")

    def test_prbench_reference_columns_are_collected_by_generic_reference_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(td) / "prbench_legal_hard_europe.jsonl"
            source_path.write_text(
                (
                    '{"task":"t3","field":"Legal","topic":"T","expert":"Non-Expert","rubric":[{"id":"c1","title":"Crit"}],'
                    '"prompt_0":"Question","response_0":"","reference_texts_0":["Doc A"],"other_reference_0":["Doc B"]}'
                    "\n"
                ),
                encoding="utf-8",
            )

            rows, report = _rows_from_source(source_path)
            self.assertEqual(report.rows_invalid, 0)
            self.assertEqual(report.rows_emitted, 1)
            first_user = rows[0]["messages"][0]["content"]
            self.assertIn("Reference Text 0:\nDoc A", first_user)
            self.assertIn("Reference Text 1:\nDoc B", first_user)

