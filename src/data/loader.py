from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..config import DatasetConfig
from ..types import LLMMessage, NormalizedExample
from .attachments import extract_attachment_contents
from .schema import SCHEMA_VERSION, is_nonempty_str, iter_jsonl_with_issues, validate_canonical_row


def normalize_row(row: Dict[str, Any], dataset: DatasetConfig) -> NormalizedExample:
    errors, warnings = validate_canonical_row(row)
    if errors:
        raise ValueError(
            f"Invalid canonical row for dataset='{dataset.name}', id='{row.get('id')}': "
            + "; ".join(errors)
        )
    if warnings:
        # Canonical loader is strict enough for required/forbidden fields;
        # keep warnings non-fatal for now.
        pass

    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version='{row.get('schema_version')}', expected '{SCHEMA_VERSION}'."
        )

    task_type = str(row["task_type"])
    judge_mode = {
        "rubric_qa": "rubric",
        "reference_qa": "reference",
        "mcq": "mcq",
    }[task_type]

    prompt = str(row.get("prompt", "")).strip()
    context = str(row.get("context", "")).strip()

    rubric = row.get("rubric") if isinstance(row.get("rubric"), list) else None
    ref_answers = row.get("reference_answers")
    reference_answer = None
    if isinstance(ref_answers, list):
        vals = [str(x).strip() for x in ref_answers if is_nonempty_str(x)]
        if vals:
            reference_answer = "\n".join(vals)

    metadata: Dict[str, Any] = {}
    if isinstance(row.get("metadata"), dict):
        metadata.update(row["metadata"])
    if isinstance(row.get("attachments"), list):
        metadata["attachments"] = row["attachments"]
        if str(metadata.get("policy_id", "")).strip() == "apexv1_extended_v1":
            metadata["attachment_contents"] = extract_attachment_contents(
                row["attachments"], dataset.path
            )

    instructions = prompt
    if task_type == "mcq":
        choices = row.get("choices") if isinstance(row.get("choices"), list) else []
        choice_lines = []
        choice_map: Dict[str, str] = {}
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_id = str(choice.get("id", "")).strip()
            choice_text = str(choice.get("text", "")).strip()
            if not choice_id or not choice_text:
                continue
            choice_map[choice_id] = choice_text
            choice_lines.append(f"{choice_id}. {choice_text}")

        if choice_lines:
            instructions = (
                f"{prompt}\n\nChoices:\n"
                + "\n".join(choice_lines)
                + "\n\nAnswer with the best option and brief reasoning."
            )

        correct_ids = row.get("correct_choice_ids")
        if isinstance(correct_ids, list):
            correct_ids_clean = [str(x).strip() for x in correct_ids if is_nonempty_str(x)]
            metadata["correct_choice_ids"] = correct_ids_clean
            if correct_ids_clean:
                parts = [
                    f"{cid}. {choice_map.get(cid, '')}".strip()
                    for cid in correct_ids_clean
                ]
                reference_answer = "\n".join(parts).strip()
        metadata["choices"] = choice_map

    return NormalizedExample(
        id=str(row["id"]),
        dataset_name=dataset.name,
        provenance=f"canonical:{task_type}",
        judge_mode=judge_mode,
        instructions=instructions,
        context=context,
        reference_answer=reference_answer,
        rubric=rubric,
        metadata=metadata,
        messages=[LLMMessage(role="user", content=instructions)],
    )


def load_examples(dataset: DatasetConfig) -> List[NormalizedExample]:
    path = Path(dataset.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset.path}")

    examples: List[NormalizedExample] = []
    for line_no, row, issue in iter_jsonl_with_issues(str(path)):
        if issue is not None:
            raise ValueError(
                f"Invalid JSON in dataset file '{dataset.path}' at line {issue.line}: "
                f"{issue.error}"
            )
        assert row is not None
        if not isinstance(row, dict):
            raise ValueError(
                f"Invalid JSON row in dataset file '{dataset.path}' at line {line_no}: "
                "top-level JSON value must be an object."
            )
        try:
            examples.append(normalize_row(row, dataset))
        except ValueError as exc:
            raise ValueError(
                f"{exc} (dataset file '{dataset.path}', line {line_no})"
            ) from exc
    if dataset.split_field and dataset.split_value is not None:
        examples = [
            e
            for e in examples
            if str(e.metadata.get(dataset.split_field)) == str(dataset.split_value)
        ]
    if dataset.limit is not None:
        examples = examples[: max(0, int(dataset.limit))]
    return examples
