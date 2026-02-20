from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ResponseKey = tuple[str, str]


def _add_response_mapping_row(
    out: dict[ResponseKey, str],
    row: dict[str, Any],
    line_no: int,
    source_label: str,
) -> None:
    example_id = str(row.get("example_id", "")).strip()
    candidate_name = str(row.get("candidate_name", "")).strip()
    response_text = row.get("response_text")
    if not example_id or not candidate_name:
        raise ValueError(
            f"Invalid {source_label} row at line {line_no}: "
            "`example_id` and `candidate_name` are required."
        )
    if not isinstance(response_text, str):
        raise ValueError(
            f"Invalid {source_label} row at line {line_no}: `response_text` must be a string."
        )
    key = (example_id, candidate_name)
    if key in out:
        duplicate_label = source_label if source_label.endswith("response") else f"{source_label} response"
        raise ValueError(
            f"Duplicate {duplicate_label} for example_id={example_id}, "
            f"candidate_name={candidate_name} at line {line_no}."
        )
    out[key] = response_text


def _load_response_mapping_jsonl(path: Path, source_label: str) -> dict[ResponseKey, str]:
    out: dict[ResponseKey, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                plural_label = (
                    "prefilled responses" if source_label == "prefilled response" else source_label
                )
                raise ValueError(
                    f"Invalid JSON in {plural_label} at line {line_no}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Invalid {source_label} row at line {line_no}: must be an object."
                )
            _add_response_mapping_row(out, row, line_no, source_label)
    return out


def _load_response_mapping_json(
    path: Path,
    source_label: str,
    candidate_names: list[str] | None = None,
) -> dict[ResponseKey, str]:
    with path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    out: dict[ResponseKey, str] = {}
    if isinstance(parsed, list):
        for idx, row in enumerate(parsed, start=1):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Invalid {source_label} row at index {idx}: must be an object."
                )
            _add_response_mapping_row(out, row, idx, source_label)
        return out

    if isinstance(parsed, dict) and all(isinstance(v, str) for v in parsed.values()):
        if source_label != "previous output":
            raise ValueError(
                f"Unsupported {source_label} JSON format. Expected list of objects with "
                "example_id/candidate_name/response_text."
            )
        names = candidate_names or []
        if len(names) != 1:
            raise ValueError(
                "Ambiguous previous output JSON mapping: task_id->response_text can only "
                "be used with exactly one configured candidate."
            )
        candidate_name = names[0]
        for example_id, response_text in parsed.items():
            out[(str(example_id), candidate_name)] = str(response_text)
        return out

    if source_label == "previous output":
        raise ValueError(
            "Unsupported previous output JSON format. Expected list of objects with "
            "example_id/candidate_name/response_text or simple mapping task_id->response_text."
        )
    raise ValueError(
        f"Unsupported {source_label} JSON format. Expected list of objects with "
        "example_id/candidate_name/response_text."
    )


def load_prefilled_responses(path: str) -> dict[ResponseKey, str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Prefilled responses file not found: {path}")
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_response_mapping_jsonl(file_path, "prefilled response")
    if suffix == ".json":
        return _load_response_mapping_json(file_path, "prefilled response")
    raise ValueError(
        f"Unsupported prefilled responses file extension '{file_path.suffix}'. Expected .jsonl or .json."
    )


def load_previous_output_responses(path: str, candidate_names: list[str]) -> dict[ResponseKey, str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Previous output file not found: {path}")
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_response_mapping_jsonl(file_path, "previous output")
    if suffix == ".json":
        return _load_response_mapping_json(
            file_path,
            "previous output",
            candidate_names=candidate_names,
        )
    raise ValueError(
        f"Unsupported previous output file extension '{file_path.suffix}'. Expected .jsonl or .json."
    )
