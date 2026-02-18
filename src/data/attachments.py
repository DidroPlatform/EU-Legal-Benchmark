from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def resolve_attachment_path(attachment_path: str, dataset_path: str) -> Path | None:
    raw = str(attachment_path).strip()
    if not raw:
        return None

    path = Path(raw)
    if path.is_absolute():
        return path if path.exists() else None

    dataset_file = Path(dataset_path).resolve()
    dataset_dir = dataset_file.parent

    candidates = [
        dataset_dir / path,
        dataset_dir.parent / path,
        dataset_dir.parent / "curated" / path,
        Path.cwd() / path,
        Path.cwd() / "data" / path,
        Path.cwd() / "data" / "curated" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages).strip()


def _extract_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def extract_attachment_contents(
    attachments: List[Dict[str, Any]], dataset_path: str
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue

        raw_path = str(item.get("path", "")).strip()
        if not raw_path:
            continue
        kind = str(item.get("kind", "")).strip().lower()

        resolved = resolve_attachment_path(raw_path, dataset_path)
        if resolved is None:
            out.append(
                {
                    "path": raw_path,
                    "kind": kind or "file",
                    "text": "",
                    "error": "Attachment file not found on disk.",
                }
            )
            continue

        try:
            if kind == "pdf" or resolved.suffix.lower() == ".pdf":
                text = _extract_pdf_text(resolved)
            else:
                text = _extract_text_file(resolved)
            out.append(
                {
                    "path": raw_path,
                    "kind": kind or ("pdf" if resolved.suffix.lower() == ".pdf" else "file"),
                    "text": text,
                    "resolved_path": str(resolved),
                }
            )
        except Exception as exc:
            out.append(
                {
                    "path": raw_path,
                    "kind": kind or "file",
                    "text": "",
                    "error": f"Failed to parse attachment: {exc}",
                    "resolved_path": str(resolved),
                }
            )
    return out
