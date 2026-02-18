from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from .schema import SCHEMA_VERSION, iter_jsonl_with_issues, validate_canonical_row


@dataclass
class BuildReport:
    source: str
    rows_read: int = 0
    rows_emitted: int = 0
    rows_invalid: int = 0
    rows_skipped: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _build_top_level(
    *,
    example_id: str,
    dataset: str,
    task_type: str,
    prompt: str,
    context: str = "",
    metadata: Mapping[str, Any] | None = None,
    attachments: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": example_id,
        "dataset": dataset,
        "task_type": task_type,
        "prompt": prompt,
    }
    if context:
        row["context"] = context
    if metadata:
        row["metadata"] = dict(metadata)
    if attachments:
        row["attachments"] = attachments
    return row


def _extract_weight(annotations: Any) -> float | None:
    if not isinstance(annotations, dict):
        return None
    for key in (
        "critically_important_weight",
        "important_weight",
        "slightly_important_weight",
        "detrimental_weight",
        "slightly_detrimental_weight",
        "critically_detrimental_weight",
    ):
        value = annotations.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _convert_prbench_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    prompts = [
        str(row.get(f"prompt_{i}", "")).strip()
        for i in range(10)
        if isinstance(row.get(f"prompt_{i}"), str)
        and str(row.get(f"prompt_{i}")).strip()
    ]
    prompt = prompts[-1] if prompts else ""

    context_chunks: List[str] = []
    for i in range(10):
        refs = row.get(f"reference_texts_{i}")
        if isinstance(refs, list):
            snippets = [
                str(x).strip() for x in refs if isinstance(x, str) and x.strip()
            ]
            if snippets:
                context_chunks.append(
                    f"Reference texts turn {i}: " + "\n\n".join(snippets)
                )

    rubric_items = row.get("rubric") if isinstance(row.get("rubric"), list) else []
    rubric: List[Dict[str, Any]] = []
    for idx, item in enumerate(rubric_items):
        if not isinstance(item, dict):
            continue
        criterion = {
            "id": str(item.get("id") or f"criterion_{idx + 1}"),
            "title": str(item.get("title") or f"Criterion {idx + 1}").strip(),
        }
        annotations = item.get("annotations")
        if isinstance(annotations, dict):
            description = annotations.get("criteria_description")
            if isinstance(description, str) and description.strip():
                criterion["description"] = description.strip()
            weight = _extract_weight(annotations)
            if weight is not None:
                criterion["weight"] = weight
        rubric.append(criterion)

    example_id = str(
        row.get("task")
        or row.get("id")
        or row.get("canary")
        or _stable_hash(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
    )

    out = _build_top_level(
        example_id=f"prbench:{example_id}",
        dataset="prbench",
        task_type="rubric_qa",
        prompt=prompt,
        context="\n\n".join(context_chunks),
        metadata={
            "topic": row.get("topic"),
            "field": row.get("field"),
            "expert": row.get("expert"),
            "category": row.get("category"),
            "decision_type": row.get("decision_type"),
            "economic_pathway": row.get("economic_pathway"),
            "classified_countries": row.get("classified_countries"),
            "source_id": example_id,
            "policy_id": "prbench_v1",
        },
    )
    out["rubric"] = rubric
    return out


def _convert_apex_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    prompt = str(row.get("Prompt", "")).strip()
    task_id = str(
        row.get("Task ID")
        or _stable_hash(prompt or json.dumps(dict(row), sort_keys=True))
    )

    raw_rubric = row.get("Rubric JSON")
    parsed_rubric = {}
    if isinstance(raw_rubric, str) and raw_rubric.strip():
        parsed_rubric = json.loads(raw_rubric)
    elif isinstance(raw_rubric, dict):
        parsed_rubric = raw_rubric

    rubric: List[Dict[str, Any]] = []
    for key, value in parsed_rubric.items():
        if not isinstance(value, dict):
            continue
        title = str(value.get("description") or key).strip()
        criterion = {
            "id": str(key).strip() or f"criterion_{len(rubric) + 1}",
            "title": title,
        }
        justification = value.get("justification")
        if isinstance(justification, str) and justification.strip():
            criterion["description"] = justification.strip()
        weight_raw = value.get("weight")
        if isinstance(weight_raw, (int, float)):
            criterion["weight"] = float(weight_raw)
        rubric.append(criterion)

    attachments: List[Dict[str, Any]] = []
    raw_attachment = row.get("File Attachments")
    if isinstance(raw_attachment, str) and raw_attachment.strip():
        attachment_path = raw_attachment.strip()
        kind = "pdf" if attachment_path.lower().endswith(".pdf") else "file"
        attachments.append({"path": attachment_path, "kind": kind})

    out = _build_top_level(
        example_id=f"apexv1:{task_id}",
        dataset="apexv1",
        task_type="rubric_qa",
        prompt=prompt,
        context="",
        attachments=attachments,
        metadata={
            "domain": row.get("Domain"),
            "source_task_id": row.get("Task ID"),
            "policy_id": "apexv1_extended_v1",
        },
    )
    out["rubric"] = rubric
    return out


def _parse_lexam_choices(raw_choices: Any) -> List[str]:
    if isinstance(raw_choices, list):
        return [str(x) for x in raw_choices]
    if isinstance(raw_choices, str) and raw_choices.strip():
        parsed = ast.literal_eval(raw_choices)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    return []


def _lexam_choice_id(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 0 <= index < len(alphabet):
        return alphabet[index]
    return f"CHOICE_{index}"


def _convert_lexam_row(row: Mapping[str, Any], line_no: int) -> Dict[str, Any]:
    source_id = str(row.get("id") or f"line_{line_no}")
    row_id = f"{source_id}:{line_no}"
    prompt = str(row.get("question", "")).strip()
    question_type = str(row.get("question_type", "")).strip().lower()

    metadata = {
        "course": row.get("course"),
        "language": row.get("language"),
        "area": row.get("area"),
        "jurisdiction": row.get("jurisdiction"),
        "year": row.get("year"),
        "question_type": question_type or None,
        "source_id": source_id,
        "policy_id": "lexam_oq_v1" if question_type == "open" else "lexam_mcq_v1",
    }

    if question_type == "open":
        answer = row.get("answer")
        if not (isinstance(answer, str) and answer.strip()):
            raise ValueError("LEXAM open question missing non-empty `answer`.")
        out = _build_top_level(
            example_id=f"lexam:open:{row_id}",
            dataset="lexam",
            task_type="reference_qa",
            prompt=prompt,
            metadata=metadata,
        )
        out["reference_answers"] = [answer.strip()]
        return out

    choices = _parse_lexam_choices(row.get("choices"))
    if len(choices) < 2:
        raise ValueError("LEXAM mcq row missing parseable `choices` with >=2 options.")

    gold = row.get("gold")
    if not isinstance(gold, (int, float)):
        raise ValueError("LEXAM mcq row missing numeric `gold`.")

    gold_index = int(gold)
    if gold_index < 0 or gold_index >= len(choices):
        raise ValueError(f"LEXAM mcq `gold` index out of range: {gold_index}.")

    choice_objects = [
        {"id": _lexam_choice_id(idx), "text": text.strip()}
        for idx, text in enumerate(choices)
    ]
    correct_choice_id = _lexam_choice_id(gold_index)

    out = _build_top_level(
        example_id=f"lexam:mcq:{row_id}",
        dataset="lexam",
        task_type="mcq",
        prompt=prompt,
        metadata={**metadata, "gold_index": gold_index},
    )
    out["choices"] = choice_objects
    out["correct_choice_ids"] = [correct_choice_id]
    return out


def _convert_includebase_row(row: Mapping[str, Any], line_no: int) -> Dict[str, Any]:
    prompt = str(row.get("question", "")).strip()
    source_file = str(row.get("source_file", "unknown"))
    example_id_seed = f"{source_file}:{line_no}:{prompt}"

    raw_choices = [
        ("A", row.get("option_a")),
        ("B", row.get("option_b")),
        ("C", row.get("option_c")),
        ("D", row.get("option_d")),
    ]
    choices_with_source_index = [
        {"source_index": idx, "id": cid, "text": str(text).strip()}
        for idx, (cid, text) in enumerate(raw_choices)
        if isinstance(text, str) and text.strip()
    ]
    if len(choices_with_source_index) < 2:
        raise ValueError("Include-base row must contain at least 2 non-empty options.")

    answer_raw = row.get("answer")
    if isinstance(answer_raw, int):
        answer_index = answer_raw
    elif isinstance(answer_raw, float) and answer_raw.is_integer():
        answer_index = int(answer_raw)
    else:
        raise ValueError("Include-base row missing integer `answer`.")

    if answer_index < 0 or answer_index >= len(raw_choices):
        raise ValueError(f"Include-base `answer` index out of range: {answer_index}.")

    gold_choice = next(
        (choice for choice in choices_with_source_index if choice["source_index"] == answer_index),
        None,
    )
    if gold_choice is None:
        raise ValueError(
            f"Include-base `answer` points to missing/empty option index {answer_index}."
        )

    choices = [{"id": c["id"], "text": c["text"]} for c in choices_with_source_index]

    out = _build_top_level(
        example_id=f"includebase:{_stable_hash(example_id_seed)}",
        dataset="includebase",
        task_type="mcq",
        prompt=prompt,
        metadata={
            "language": row.get("language"),
            "country": row.get("country"),
            "subject": row.get("subject"),
            "level": row.get("level"),
            "is_european": row.get("is_european"),
            "source_file": source_file,
            "answer_index": answer_index,
            "policy_id": "includebase_default_v1",
        },
    )
    out["choices"] = choices
    out["correct_choice_ids"] = [gold_choice["id"]]
    return out


def _convert_lar_echr_row(row: Mapping[str, Any], line_no: int) -> Dict[str, Any]:
    facts = str(row.get("facts", "")).strip()
    arguments = str(row.get("context_arguments") or row.get("context") or "").strip()
    if not facts:
        raise ValueError("LAR-ECHR row missing non-empty `facts`.")
    if not arguments:
        raise ValueError("LAR-ECHR row missing non-empty `context`/`context_arguments`.")

    raw_choices = [
        ("A", row.get("a")),
        ("B", row.get("b")),
        ("C", row.get("c")),
        ("D", row.get("d")),
    ]
    choices = [
        {"id": cid, "text": str(text).strip()}
        for cid, text in raw_choices
        if isinstance(text, str) and text.strip()
    ]
    if len(choices) < 2:
        raise ValueError("LAR-ECHR row must contain at least 2 non-empty choices.")

    label = str(row.get("label", "")).strip().upper()
    if not label:
        raise ValueError("LAR-ECHR row missing non-empty `label`.")
    valid_ids = {c["id"] for c in choices}
    if label not in valid_ids:
        raise ValueError(
            f"LAR-ECHR `label` '{label}' not present in available choices {sorted(valid_ids)}."
        )

    case_id = str(row.get("case_id", "")).strip()
    source_id = str(row.get("record_id") or case_id or f"line_{line_no}")
    if case_id:
        example_id = f"lar_echr:{case_id}"
    else:
        seed = f"{line_no}:{facts}:{arguments}"
        example_id = f"lar_echr:{_stable_hash(seed)}"

    metadata: Dict[str, Any] = {
        "policy_id": "lar_echr_mcq_v1",
        "source_id": source_id,
        "case_id": case_id or None,
        "case_no": row.get("case_no"),
        "source_split": row.get("source_split"),
        "source_dataset": row.get("source_dataset"),
    }
    for key in (
        "toughness_score",
        "difficulty_signal_facts_len",
        "difficulty_signal_context_len",
        "difficulty_signal_option_similarity",
        "difficulty_signal_overlap_ambiguity",
        "difficulty_signal_negation_density",
    ):
        value = row.get(key)
        if isinstance(value, (int, float)):
            metadata[key] = float(value)

    out = _build_top_level(
        example_id=example_id,
        dataset="lar_echr",
        task_type="mcq",
        prompt="Select the continuation that best extends the ECHR argument excerpt.",
        context=f"Facts:\n{facts}\n\nArguments:\n{arguments}",
        metadata=metadata,
    )
    out["choices"] = choices
    out["correct_choice_ids"] = [label]
    return out


def _rows_from_source(path: Path) -> Tuple[List[Dict[str, Any]], BuildReport]:
    report = BuildReport(source=str(path))
    emitted: List[Dict[str, Any]] = []
    source_name = path.name

    for line_no, row, issue in iter_jsonl_with_issues(path):
        report.rows_read += 1
        if issue is not None:
            report.rows_invalid += 1
            report.errors.append(
                {
                    "line": line_no,
                    "id": None,
                    "error": issue.error,
                }
            )
            continue
        assert row is not None
        if not isinstance(row, dict):
            report.rows_invalid += 1
            report.errors.append(
                {
                    "line": line_no,
                    "id": None,
                    "error": "Top-level JSON value must be an object.",
                }
            )
            continue
        try:
            if source_name == "prbench_legal_hard_europe.jsonl":
                converted = _convert_prbench_row(row)
            elif source_name == "apexv1_legal_europe.jsonl":
                converted = _convert_apex_row(row)
            elif source_name == "lexam_swiss_tough_diverse.jsonl":
                converted = _convert_lexam_row(row, line_no)
            elif source_name == "includebase_europe_law.jsonl":
                converted = _convert_includebase_row(row, line_no)
            elif source_name == "lar_echr_tough_17.jsonl":
                converted = _convert_lar_echr_row(row, line_no)
            else:
                report.rows_skipped += 1
                report.errors.append(
                    {
                        "line": line_no,
                        "id": row.get("id"),
                        "error": "Unsupported curated file.",
                    }
                )
                continue
        except Exception as exc:
            report.rows_invalid += 1
            report.errors.append(
                {
                    "line": line_no,
                    "id": row.get("id"),
                    "error": f"Conversion error: {exc}",
                }
            )
            continue

        errors, warnings = validate_canonical_row(converted)
        if errors:
            report.rows_invalid += 1
            report.errors.append(
                {
                    "line": line_no,
                    "id": converted.get("id"),
                    "error": "Schema validation failed.",
                    "details": errors,
                }
            )
            continue
        if warnings:
            report.errors.append(
                {
                    "line": line_no,
                    "id": converted.get("id"),
                    "warning": "Schema validation warning.",
                    "details": warnings,
                }
            )
        report.rows_emitted += 1
        emitted.append(converted)

    return emitted, report


def build_merged_eval_file(sources: List[Path], output_jsonl: Path) -> Dict[str, Any]:
    merged_rows: List[Dict[str, Any]] = []
    reports: List[BuildReport] = []
    seen_ids = set()
    duplicate_id_errors: List[Dict[str, Any]] = []

    for source in sources:
        rows, report = _rows_from_source(source)
        for row in rows:
            row_id = row["id"]
            if row_id in seen_ids:
                report.rows_invalid += 1
                report.rows_emitted -= 1
                duplicate_id_errors.append(
                    {
                        "source": str(source),
                        "id": row_id,
                        "error": "Duplicate id in merged output.",
                    }
                )
                continue
            seen_ids.add(row_id)
            merged_rows.append(row)
        reports.append(report)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "output_jsonl": str(output_jsonl),
        "sources": [str(s) for s in sources],
        "rows_written": len(merged_rows),
        "unique_ids": len(seen_ids),
        "duplicate_id_errors": duplicate_id_errors,
        "reports": [
            {
                "source": r.source,
                "rows_read": r.rows_read,
                "rows_emitted": r.rows_emitted,
                "rows_invalid": r.rows_invalid,
                "rows_skipped": r.rows_skipped,
                "messages": r.errors,
            }
            for r in reports
        ],
        "task_type_breakdown": {
            "rubric_qa": sum(1 for r in merged_rows if r["task_type"] == "rubric_qa"),
            "reference_qa": sum(
                1 for r in merged_rows if r["task_type"] == "reference_qa"
            ),
            "mcq": sum(1 for r in merged_rows if r["task_type"] == "mcq"),
        },
    }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert curated datasets into one merged legal_eval_v1 JSONL file."
    )
    parser.add_argument(
        "--sources-dir",
        default="data/curated",
        help="Directory containing curated source JSONL files.",
    )
    parser.add_argument(
        "--output",
        default="data/for_eval/merged_legal_eval_v1.jsonl",
        help="Output merged canonical JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.sources_dir)

    # Explicit source order to keep deterministic output across runs.
    source_names = [
        "prbench_legal_hard_europe.jsonl",
        "apexv1_legal_europe.jsonl",
        "lexam_swiss_tough_diverse.jsonl",
        "includebase_europe_law.jsonl",
        "lar_echr_tough_17.jsonl",
    ]
    sources = [
        source_dir / name for name in source_names if (source_dir / name).exists()
    ]
    if not sources:
        raise FileNotFoundError(f"No supported curated files found in {source_dir}.")

    summary = build_merged_eval_file(
        sources=sources,
        output_jsonl=Path(args.output),
    )
    print(f"Wrote {summary['rows_written']} rows to {summary['output_jsonl']}")
    print("Task type breakdown:", summary["task_type_breakdown"])


if __name__ == "__main__":
    main()
