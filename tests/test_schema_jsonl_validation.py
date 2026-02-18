from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.schema import validate_jsonl_file


class TestSchemaJsonlValidation(unittest.TestCase):
    def test_malformed_json_line_is_reported_and_validation_continues(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mixed.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"schema_version":"legal_eval_v1","id":"ok-1","dataset":"d","task_type":"reference_qa","prompt":"P","reference_answers":["A"]}',
                        '{"schema_version":"legal_eval_v1","id":"broken"',
                        '{"schema_version":"legal_eval_v1","id":"ok-2","dataset":"d","task_type":"reference_qa","prompt":"P","reference_answers":["B"]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_jsonl_file(path)
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["valid_rows"], 2)
            self.assertEqual(report["invalid_rows"], 1)
            self.assertEqual(report["errors"][0]["line"], 2)
            self.assertIn("JSON parse error", report["errors"][0]["errors"][0])


if __name__ == "__main__":
    unittest.main()
