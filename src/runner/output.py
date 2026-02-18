from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.config import BenchmarkConfig, ModelConfig


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _merge_scored_rows(
    responses: List[Dict[str, Any]], judgments: List[Dict[str, Any]], run_started_at_utc: str
) -> List[Dict[str, Any]]:
    judgment_by_key = {
        (j["dataset"], j["example_id"], j["candidate_name"]): j for j in judgments
    }
    merged: List[Dict[str, Any]] = []
    for response in responses:
        key = (response["dataset"], response["example_id"], response["candidate_name"])
        judgment = judgment_by_key.get(key)
        row = dict(response)
        row["run_started_at_utc"] = run_started_at_utc
        if judgment is None:
            row["grading"] = None
        else:
            row["grading"] = {
                "judge_name": judgment.get("judge_name"),
                "judge_provider": judgment.get("judge_provider"),
                "judge_model": judgment.get("judge_model"),
                "judge_settings": judgment.get("judge_settings"),
                "judge_request_id": judgment.get("request_id"),
                "judge_cache_key": judgment.get("cache_key"),
                "judge_cache_hit": judgment.get("cache_hit"),
                "score": judgment.get("score"),
                "pass": judgment.get("pass"),
                "rationale": judgment.get("rationale"),
                "criteria": judgment.get("criteria"),
                "parse_error": judgment.get("parse_error"),
                "raw_judge": judgment.get("raw_judge"),
            }
        merged.append(row)
    return merged


def _summary(responses: List[Dict[str, Any]], judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, Dict[str, Any]] = {}
    by_dataset_model: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for response in responses:
        model_name = response["candidate_name"]
        dataset_name = response["dataset"]
        by_model.setdefault(model_name, {"responses": 0, "judged": 0, "score_sum": 0.0, "pass_count": 0})
        by_model[model_name]["responses"] += 1

        by_dataset_model.setdefault(dataset_name, {})
        by_dataset_model[dataset_name].setdefault(
            model_name,
            {"responses": 0, "judged": 0, "score_sum": 0.0, "pass_count": 0},
        )
        by_dataset_model[dataset_name][model_name]["responses"] += 1

    for judgment in judgments:
        model_name = judgment["candidate_name"]
        dataset_name = judgment["dataset"]

        by_model.setdefault(model_name, {"responses": 0, "judged": 0, "score_sum": 0.0, "pass_count": 0})
        by_model[model_name]["judged"] += 1
        by_model[model_name]["score_sum"] += float(judgment["score"])
        by_model[model_name]["pass_count"] += 1 if bool(judgment["pass"]) else 0

        by_dataset_model.setdefault(dataset_name, {})
        by_dataset_model[dataset_name].setdefault(
            model_name,
            {"responses": 0, "judged": 0, "score_sum": 0.0, "pass_count": 0},
        )
        by_dataset_model[dataset_name][model_name]["judged"] += 1
        by_dataset_model[dataset_name][model_name]["score_sum"] += float(judgment["score"])
        by_dataset_model[dataset_name][model_name]["pass_count"] += 1 if bool(judgment["pass"]) else 0

    for stats in by_model.values():
        judged = max(1, stats["judged"])
        stats["avg_score"] = stats["score_sum"] / judged
        stats["pass_rate"] = stats["pass_count"] / judged

    for dataset_stats in by_dataset_model.values():
        for stats in dataset_stats.values():
            judged = max(1, stats["judged"])
            stats["avg_score"] = stats["score_sum"] / judged
            stats["pass_rate"] = stats["pass_count"] / judged

    return {
        "models": by_model,
        "by_dataset": by_dataset_model,
        "num_responses": len(responses),
        "num_judgments": len(judgments),
    }


def write_run_outputs(
    *,
    output_dir: Path,
    config: BenchmarkConfig,
    run_id: str,
    run_started_at_utc: str,
    examples: List[Any],
    dataset_stats: List[Dict[str, Any]],
    normalized_rows: List[Dict[str, Any]],
    responses_rows: List[Dict[str, Any]],
    judgments_rows: List[Dict[str, Any]],
    trace_rows: List[Dict[str, Any]],
    failed_items: List[Dict[str, Any]],
    run_status: str,
    interrupted_stage: str | None,
    judge_descriptor: Any,
) -> Dict[str, Any]:
    _write_jsonl(output_dir / "examples.jsonl", normalized_rows)
    _write_jsonl(output_dir / "responses.jsonl", responses_rows)
    _write_jsonl(output_dir / "judgments.jsonl", judgments_rows)
    _write_jsonl(
        output_dir / "scored_responses.jsonl",
        _merge_scored_rows(
            responses=responses_rows,
            judgments=judgments_rows,
            run_started_at_utc=run_started_at_utc,
        ),
    )
    _write_jsonl(output_dir / "trace.jsonl", trace_rows)

    summary = _summary(responses_rows, judgments_rows)
    summary.update(
        {
            "run_id": run_id,
            "run_started_at_utc": run_started_at_utc,
            "selected_examples": len(examples),
            "datasets": dataset_stats,
            "judge": judge_descriptor(config.judge),
            "judges": [judge_descriptor(j) for j in config.judges],
            "failed_items": failed_items,
            "num_failures": len(failed_items),
            "run_status": run_status,
            "interrupted_stage": interrupted_stage,
        }
    )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "providers": sorted(list(config.providers.keys())),
                "candidates": [c.__dict__ for c in config.candidates],
                "judge": config.judge.__dict__,
                "judges": [j.__dict__ for j in config.judges],
                "datasets": [d.__dict__ for d in config.data.datasets],
                "retry": config.retry.__dict__,
                "cache": config.cache.__dict__,
                "run": config.run.__dict__,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return summary
