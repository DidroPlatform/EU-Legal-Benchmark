from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple


SCHEMA_VERSION = "legal_eval_v1"
TASK_TYPES = {"rubric_qa", "reference_qa", "mcq"}

COMMON_REQUIRED_FIELDS = {"schema_version", "id", "dataset", "task_type", "prompt"}
COMMON_OPTIONAL_FIELDS = {
    "context",
    "messages",
    "attachments",
    "metadata",
    "rubric",
    "reference_answers",
    "choices",
    "correct_choice_ids",
}


@dataclass(frozen=True)
class JsonlParseIssue:
    line: int
    error: str
    raw_line: str = ""


def is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_attachments(attachments: Any, errors: List[str]) -> None:
    if attachments is None:
        return
    if not isinstance(attachments, list):
        errors.append("`attachments` must be an array when provided.")
        return
    for i, item in enumerate(attachments):
        if not isinstance(item, dict):
            errors.append(f"`attachments[{i}]` must be an object.")
            continue
        if not is_nonempty_str(item.get("path")):
            errors.append(f"`attachments[{i}].path` must be a non-empty string.")
        for key in ("kind", "title"):
            if key in item and not isinstance(item[key], str):
                errors.append(
                    f"`attachments[{i}].{key}` must be a string when provided."
                )


def _validate_messages(messages: Any, errors: List[str]) -> None:
    if messages is None:
        return
    if not isinstance(messages, list):
        errors.append("`messages` must be an array when provided.")
        return
    allowed_roles = {"user", "assistant", "system"}
    for i, item in enumerate(messages):
        if not isinstance(item, dict):
            errors.append(f"`messages[{i}]` must be an object.")
            continue
        role = item.get("role")
        content = item.get("content")
        if not is_nonempty_str(role):
            errors.append(f"`messages[{i}].role` must be a non-empty string.")
        elif str(role) not in allowed_roles:
            errors.append(
                f"`messages[{i}].role` must be one of: assistant, system, user."
            )
        if not is_nonempty_str(content):
            errors.append(f"`messages[{i}].content` must be a non-empty string.")


def _validate_rubric(rubric: Any, errors: List[str]) -> None:
    if not isinstance(rubric, list) or not rubric:
        errors.append("`rubric` must be a non-empty array.")
        return
    for i, criterion in enumerate(rubric):
        if not isinstance(criterion, dict):
            errors.append(f"`rubric[{i}]` must be an object.")
            continue
        if not is_nonempty_str(criterion.get("id")):
            errors.append(f"`rubric[{i}].id` must be a non-empty string.")
        if not is_nonempty_str(criterion.get("title")):
            errors.append(f"`rubric[{i}].title` must be a non-empty string.")
        if "description" in criterion and not isinstance(criterion["description"], str):
            errors.append(f"`rubric[{i}].description` must be a string when provided.")
        if "weight" in criterion and not isinstance(criterion["weight"], (int, float)):
            errors.append(f"`rubric[{i}].weight` must be a number when provided.")


def _validate_reference_answers(reference_answers: Any, errors: List[str]) -> None:
    if not isinstance(reference_answers, list) or not reference_answers:
        errors.append("`reference_answers` must be a non-empty array.")
        return
    for i, value in enumerate(reference_answers):
        if not is_nonempty_str(value):
            errors.append(f"`reference_answers[{i}]` must be a non-empty string.")


def _validate_mcq_fields(
    choices: Any, correct_choice_ids: Any, errors: List[str]
) -> None:
    if not isinstance(choices, list) or len(choices) < 2:
        errors.append("`choices` must be an array with at least 2 elements.")
        return

    seen_ids = set()
    for i, choice in enumerate(choices):
        if not isinstance(choice, dict):
            errors.append(f"`choices[{i}]` must be an object.")
            continue
        choice_id = choice.get("id")
        if not is_nonempty_str(choice_id):
            errors.append(f"`choices[{i}].id` must be a non-empty string.")
        elif choice_id in seen_ids:
            errors.append(f"`choices[{i}].id` must be unique; duplicate `{choice_id}`.")
        else:
            seen_ids.add(choice_id)

        if not is_nonempty_str(choice.get("text")):
            errors.append(f"`choices[{i}].text` must be a non-empty string.")

    if not isinstance(correct_choice_ids, list) or not correct_choice_ids:
        errors.append("`correct_choice_ids` must be a non-empty array.")
        return

    seen_correct = set()
    for i, cid in enumerate(correct_choice_ids):
        if not is_nonempty_str(cid):
            errors.append(f"`correct_choice_ids[{i}]` must be a non-empty string.")
            continue
        if cid in seen_correct:
            errors.append(f"`correct_choice_ids[{i}]` duplicates choice id `{cid}`.")
            continue
        seen_correct.add(cid)
        if cid not in seen_ids:
            errors.append(
                f"`correct_choice_ids[{i}]` references unknown choice id `{cid}`."
            )


def validate_canonical_row(row: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(row, Mapping):
        return (["Row must be a JSON object."], warnings)

    missing = [field for field in COMMON_REQUIRED_FIELDS if field not in row]
    if missing:
        errors.append(f"Missing required fields: {', '.join(sorted(missing))}.")
        return (errors, warnings)

    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"`schema_version` must be `{SCHEMA_VERSION}`.")

    for field in ("id", "dataset", "prompt"):
        if not is_nonempty_str(row.get(field)):
            errors.append(f"`{field}` must be a non-empty string.")

    task_type = row.get("task_type")
    if task_type not in TASK_TYPES:
        errors.append("`task_type` must be one of: rubric_qa, reference_qa, mcq.")
        return (errors, warnings)

    if "context" in row and not isinstance(row["context"], str):
        errors.append("`context` must be a string when provided.")

    if "metadata" in row and not isinstance(row["metadata"], dict):
        errors.append("`metadata` must be an object when provided.")

    _validate_messages(row.get("messages"), errors)
    _validate_attachments(row.get("attachments"), errors)

    allowed_fields = COMMON_REQUIRED_FIELDS | COMMON_OPTIONAL_FIELDS
    unknown_fields = [key for key in row.keys() if key not in allowed_fields]
    if unknown_fields:
        warnings.append(
            "Unknown top-level fields present (consider moving into `metadata`): "
            + ", ".join(sorted(unknown_fields))
            + "."
        )

    if task_type == "rubric_qa":
        _validate_rubric(row.get("rubric"), errors)
        if "reference_answers" in row:
            if not isinstance(row["reference_answers"], list):
                errors.append("`reference_answers` must be an array when provided.")
            else:
                for i, value in enumerate(row["reference_answers"]):
                    if not is_nonempty_str(value):
                        errors.append(
                            f"`reference_answers[{i}]` must be a non-empty string."
                        )
        for forbidden in ("choices", "correct_choice_ids"):
            if forbidden in row:
                errors.append(f"`{forbidden}` is forbidden for task_type=rubric_qa.")

    elif task_type == "reference_qa":
        _validate_reference_answers(row.get("reference_answers"), errors)
        for forbidden in ("rubric", "choices", "correct_choice_ids"):
            if forbidden in row:
                errors.append(f"`{forbidden}` is forbidden for task_type=reference_qa.")

    elif task_type == "mcq":
        _validate_mcq_fields(row.get("choices"), row.get("correct_choice_ids"), errors)
        for forbidden in ("rubric", "reference_answers"):
            if forbidden in row:
                errors.append(f"`{forbidden}` is forbidden for task_type=mcq.")

    return (errors, warnings)


def iter_jsonl_with_issues(
    path: str | Path,
) -> Iterable[Tuple[int, Dict[str, Any] | None, JsonlParseIssue | None]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw_line = line.rstrip("\n")
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                yield (
                    line_no,
                    None,
                    JsonlParseIssue(
                        line=line_no,
                        error=f"JSON parse error: {exc.msg} (column {exc.colno})",
                        raw_line=raw_line,
                    ),
                )
                continue
            yield line_no, parsed, None


def validate_jsonl_file(path: str | Path) -> Dict[str, Any]:
    rows = 0
    valid_rows = 0
    invalid_rows = 0
    warning_rows = 0
    errors_by_row: List[Dict[str, Any]] = []
    warnings_by_row: List[Dict[str, Any]] = []

    for line_no, row, parse_issue in iter_jsonl_with_issues(path):
        rows += 1
        if parse_issue is not None:
            invalid_rows += 1
            errors_by_row.append(
                {
                    "line": parse_issue.line,
                    "id": None,
                    "errors": [parse_issue.error],
                }
            )
            continue
        assert row is not None
        errors, warnings = validate_canonical_row(row)
        if errors:
            invalid_rows += 1
            errors_by_row.append({"line": line_no, "id": row.get("id"), "errors": errors})
        else:
            valid_rows += 1
        if warnings:
            warning_rows += 1
            warnings_by_row.append(
                {"line": line_no, "id": row.get("id"), "warnings": warnings}
            )

    return {
        "path": str(path),
        "rows": rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "warning_rows": warning_rows,
        "errors": errors_by_row,
        "warnings": warnings_by_row,
    }
