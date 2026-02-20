from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.io.json_io import read_json, read_jsonl, write_json, write_jsonl
from src.runner.output import build_summary, merge_scored_rows
from src.runner.reconcile import overlay_rows, row_key


RowKey = Tuple[str, str, str]


def _example_row_key_fields(row: Dict[str, Any]) -> tuple[str, str] | None:
    dataset = row.get("dataset")
    example_id = row.get("example_id")
    if example_id is None:
        example_id = row.get("id")
    if not isinstance(dataset, str) or not dataset.strip():
        return None
    if not isinstance(example_id, str) or not example_id.strip():
        return None
    return dataset, example_id


def _expected_keys(
    base_examples: List[Dict[str, Any]],
    base_run_config: Dict[str, Any],
) -> set[RowKey]:
    candidate_names = [
        c.get("name")
        for c in base_run_config.get("candidates", [])
        if isinstance(c, dict) and isinstance(c.get("name"), str) and str(c.get("name")).strip()
    ]
    keys: set[RowKey] = set()
    for row in base_examples:
        fields = _example_row_key_fields(row)
        if fields is None:
            continue
        dataset, example_id = fields
        for candidate_name in candidate_names:
            keys.add((dataset, example_id, candidate_name))
    return keys


def _missing_failure_items(keys: List[RowKey], stage: str, message: str) -> List[Dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "example_id": example_id,
            "candidate_name": candidate_name,
            "stage": stage,
            "error": message,
        }
        for dataset, example_id, candidate_name in keys
    ]


def merge_run_outputs(base_run_dir: Path, backfill_run_dir: Path, out_run_dir: Path) -> Dict[str, Any]:
    base_outputs = base_run_dir / "outputs"
    patch_outputs = backfill_run_dir / "outputs"
    out_outputs = out_run_dir / "outputs"
    out_outputs.mkdir(parents=True, exist_ok=True)

    base_examples = read_jsonl(base_outputs / "examples.jsonl")
    base_responses = read_jsonl(base_outputs / "responses.jsonl")
    base_judgments = read_jsonl(base_outputs / "judgments.jsonl")
    base_trace = read_jsonl(base_outputs / "trace.jsonl")
    base_summary = read_json(base_outputs / "summary.json")
    base_run_config = read_json(base_outputs / "run_config.json")

    patch_responses = read_jsonl(patch_outputs / "responses.jsonl")
    patch_judgments = read_jsonl(patch_outputs / "judgments.jsonl")
    patch_trace = read_jsonl(patch_outputs / "trace.jsonl")

    merged_responses, replaced_responses = overlay_rows(base_responses, patch_responses)
    merged_judgments, replaced_judgments = overlay_rows(base_judgments, patch_judgments)
    merged_trace = [*base_trace, *patch_trace]

    run_started_at_utc = str(base_summary.get("run_started_at_utc") or datetime.now(timezone.utc).isoformat())
    merged_scored = merge_scored_rows(
        responses=merged_responses,
        judgments=merged_judgments,
        run_started_at_utc=run_started_at_utc,
    )

    expected_pairs = _expected_keys(base_examples, base_run_config)
    response_pairs = {row_key(row) for row in merged_responses}
    judgment_pairs = {row_key(row) for row in merged_judgments}
    missing_response_keys = sorted(expected_pairs - response_pairs)
    missing_judgment_keys = sorted(expected_pairs - judgment_pairs)
    failed_items = [
        *_missing_failure_items(
            missing_response_keys,
            stage="response_missing_after_merge",
            message="Missing response row after merge overlay.",
        ),
        *_missing_failure_items(
            missing_judgment_keys,
            stage="judge_missing_after_merge",
            message="Missing judgment row after merge overlay.",
        ),
    ]
    merged_run_status = "degraded" if failed_items else "completed"

    merged_summary = build_summary(merged_responses, merged_judgments)
    merged_summary.update(
        {
            "run_id": out_run_dir.name,
            "run_started_at_utc": run_started_at_utc,
            "selected_examples": base_summary.get("selected_examples"),
            "datasets": base_summary.get("datasets", []),
            "judges": base_summary.get("judges", []),
            "failed_items": failed_items,
            "num_failures": len(failed_items),
            "run_status": merged_run_status,
            "interrupted_stage": None,
            "repaired_from_run_id": base_run_dir.name,
            "backfill_run_id": backfill_run_dir.name,
        }
    )

    expected_total = len(expected_pairs)
    missing_responses = len(missing_response_keys)
    missing_judgments = len(missing_judgment_keys)
    report = {
        "base_run_id": base_run_dir.name,
        "backfill_run_id": backfill_run_dir.name,
        "output_run_id": out_run_dir.name,
        "replaced_responses": replaced_responses,
        "replaced_judgments": replaced_judgments,
        "added_responses": len(merged_responses) - len(base_responses),
        "added_judgments": len(merged_judgments) - len(base_judgments),
        "expected_total_pairs": expected_total,
        "missing_responses_after_merge": missing_responses,
        "missing_judgments_after_merge": missing_judgments,
    }

    write_jsonl(out_outputs / "examples.jsonl", base_examples)
    write_jsonl(out_outputs / "responses.jsonl", merged_responses)
    write_jsonl(out_outputs / "judgments.jsonl", merged_judgments)
    write_jsonl(out_outputs / "scored_responses.jsonl", merged_scored)
    write_jsonl(out_outputs / "trace.jsonl", merged_trace)
    write_json(out_outputs / "summary.json", merged_summary)
    write_json(out_outputs / "run_config.json", base_run_config)
    write_json(out_outputs / "reconciliation_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a base run with a targeted backfill run.")
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--backfill-run-id", required=True)
    parser.add_argument("--out-run-id", default=None, help="Optional output run ID. Defaults to timestamped repair ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_run_dir = Path("data/runs") / args.base_run_id
    backfill_run_dir = Path("data/runs") / args.backfill_run_id
    out_run_id = args.out_run_id or f"{args.base_run_id}_repaired_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_run_dir = Path("data/runs") / out_run_id

    report = merge_run_outputs(base_run_dir, backfill_run_dir, out_run_dir)
    print(f"Merged run written to: {out_run_dir}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
