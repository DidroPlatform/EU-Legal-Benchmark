from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.attachments import (
    extract_attachment_contents,
    resolve_attachment_path,
)


class TestAttachments(unittest.TestCase):
    def test_resolve_attachment_prefers_data_curated_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_path = root / "data" / "for_eval" / "merged.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text("", encoding="utf-8")

            attachment_rel = "documents/1/file.txt"
            file_path = root / "data" / "curated" / attachment_rel
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("hello", encoding="utf-8")

            resolved = resolve_attachment_path(
                attachment_rel,
                str(dataset_path),
            )
            self.assertEqual(resolved, file_path.resolve())

    def test_extract_attachment_contents_reads_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_path = root / "data" / "for_eval" / "merged.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text("", encoding="utf-8")

            attachment_rel = "documents/1/file.txt"
            file_path = root / "data" / "curated" / attachment_rel
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("Document content", encoding="utf-8")

            parsed = extract_attachment_contents(
                [{"path": attachment_rel, "kind": "file"}],
                str(dataset_path),
            )
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["path"], attachment_rel)
            self.assertEqual(parsed[0]["text"], "Document content")
            self.assertNotIn("error", parsed[0])

    def test_extract_attachment_contents_marks_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_path = root / "data" / "for_eval" / "merged.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text("", encoding="utf-8")

            parsed = extract_attachment_contents(
                [{"path": "documents/404/missing.pdf", "kind": "pdf"}],
                str(dataset_path),
            )
            self.assertEqual(len(parsed), 1)
            self.assertIn("error", parsed[0])
            self.assertEqual(parsed[0]["text"], "")


if __name__ == "__main__":
    unittest.main()
