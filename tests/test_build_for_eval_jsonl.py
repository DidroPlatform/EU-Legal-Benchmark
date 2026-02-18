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


if __name__ == "__main__":
    unittest.main()
