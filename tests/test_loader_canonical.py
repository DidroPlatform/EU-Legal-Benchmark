from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import DatasetConfig
from src.data.loader import load_examples


class TestCanonicalLoader(unittest.TestCase):
    def test_rejects_non_canonical_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.jsonl"
            path.write_text('{"question":"What is law?"}\n', encoding="utf-8")

            ds = DatasetConfig(name="bad", path=str(path))
            with self.assertRaises(ValueError) as ctx:
                load_examples(ds)

            message = str(ctx.exception)
            self.assertIn("Invalid canonical row", message)
            self.assertIn("Missing required fields", message)

    def test_apex_row_extracts_attachment_contents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_path = root / "data" / "for_eval" / "merged.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)

            attachment_rel = "documents/123/file.txt"
            attachment_path = root / "data" / "curated" / attachment_rel
            attachment_path.parent.mkdir(parents=True, exist_ok=True)
            attachment_path.write_text("Attachment text", encoding="utf-8")

            dataset_path.write_text(
                "\n".join(
                    [
                        '{"schema_version":"legal_eval_v1","id":"apexv1:1","dataset":"apexv1","task_type":"rubric_qa","prompt":"P","attachments":[{"path":"documents/123/file.txt","kind":"file"}],"metadata":{"policy_id":"apexv1_extended_v1"},"rubric":[{"id":"c1","title":"t1"}]}'
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            ds = DatasetConfig(name="ok", path=str(dataset_path))
            examples = load_examples(ds)
            self.assertEqual(len(examples), 1)
            attachment_contents = examples[0].metadata.get("attachment_contents")
            self.assertIsInstance(attachment_contents, list)
            self.assertEqual(attachment_contents[0]["path"], attachment_rel)
            self.assertEqual(attachment_contents[0]["text"], "Attachment text")

    def test_reports_line_number_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad-json.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"schema_version":"legal_eval_v1","id":"ok","dataset":"d","task_type":"reference_qa","prompt":"P","reference_answers":["A"]}',
                        '{"schema_version":"legal_eval_v1","id":"broken","dataset":"d"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            ds = DatasetConfig(name="bad_json", path=str(path))
            with self.assertRaises(ValueError) as ctx:
                load_examples(ds)

            message = str(ctx.exception)
            self.assertIn("Invalid JSON in dataset file", message)
            self.assertIn("line 2", message)


if __name__ == "__main__":
    unittest.main()
