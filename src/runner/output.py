from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import BenchmarkConfig, ModelConfig
from src.io.json_io import write_json, write_jsonl
from src.runner.services import JudgeDescriptorFn


@dataclass
class ModelStats:
    responses: int = 0
    judged: int = 0
    score_sum: float = 0.0
    pass_count: int = 0
    prbench_normalized_sum: float = 0.0
    prbench_normalized_count: int = 0
    prbench_clipped_sum: float = 0.0
    prbench_clipped_count: int = 0

    def add_response(self) -> None:
        self.responses += 1

    def add_judgment(
        self,
        score: float,
        passed: bool,
        prbench_normalized: Optional[float],
        prbench_clipped: Optional[float],
    ) -> None:
        self.judged += 1
        self.score_sum += score
        if passed:
            self.pass_count += 1
        if isinstance(prbench_normalized, (int, float)):
            self.prbench_normalized_sum += float(prbench_normalized)
            self.prbench_normalized_count += 1
        if isinstance(prbench_clipped, (int, float)):
            self.prbench_clipped_sum += float(prbench_clipped)
            self.prbench_clipped_count += 1

    def to_dict(self) -> Dict[str, Any]:
        judged = max(1, self.judged)
        result: Dict[str, Any] = {
            "responses": self.responses,
            "judged": self.judged,
            "score_sum": self.score_sum,
            "pass_count": self.pass_count,
            "avg_score": self.score_sum / judged,
            "pass_rate": self.pass_count / judged,
        }
        if self.prbench_normalized_count:
            result["prbench_normalized_sum"] = self.prbench_normalized_sum
            result["prbench_normalized_count"] = self.prbench_normalized_count
            result["prbench_avg_points_normalized"] = (
                self.prbench_normalized_sum / self.prbench_normalized_count
            )
        if self.prbench_clipped_count:
            result["prbench_clipped_sum"] = self.prbench_clipped_sum
            result["prbench_clipped_count"] = self.prbench_clipped_count
            result["prbench_avg_points_clipped"] = (
                self.prbench_clipped_sum / self.prbench_clipped_count
            )
        return result


def merge_scored_rows(
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
                "prbench_weighted_raw": judgment.get("prbench_weighted_raw"),
                "prbench_points_normalized": judgment.get("prbench_points_normalized"),
                "prbench_points_clipped": judgment.get("prbench_points_clipped"),
                "raw_judge": judgment.get("raw_judge"),
            }
        merged.append(row)
    return merged


def build_summary(responses: List[Dict[str, Any]], judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, ModelStats] = {}
    by_dataset_model: Dict[str, Dict[str, ModelStats]] = {}

    for response in responses:
        model_name = response["candidate_name"]
        dataset_name = response["dataset"]

        by_model.setdefault(model_name, ModelStats()).add_response()
        by_dataset_model.setdefault(dataset_name, {}).setdefault(model_name, ModelStats()).add_response()

    for judgment in judgments:
        model_name = judgment["candidate_name"]
        dataset_name = judgment["dataset"]
        score = float(judgment["score"])
        passed = bool(judgment["pass"])
        prbench_normalized = judgment.get("prbench_points_normalized")
        prbench_clipped = judgment.get("prbench_points_clipped")

        by_model.setdefault(model_name, ModelStats()).add_judgment(
            score, passed, prbench_normalized, prbench_clipped
        )
        by_dataset_model.setdefault(dataset_name, {}).setdefault(model_name, ModelStats()).add_judgment(
            score, passed, prbench_normalized, prbench_clipped
        )

    return {
        "models": {name: stats.to_dict() for name, stats in by_model.items()},
        "by_dataset": {
            dataset: {name: stats.to_dict() for name, stats in model_map.items()}
            for dataset, model_map in by_dataset_model.items()
        },
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
    judge_descriptor: JudgeDescriptorFn,
) -> Dict[str, Any]:
    write_jsonl(output_dir / "examples.jsonl", normalized_rows)
    write_jsonl(output_dir / "responses.jsonl", responses_rows)
    write_jsonl(output_dir / "judgments.jsonl", judgments_rows)
    write_jsonl(
        output_dir / "scored_responses.jsonl",
        merge_scored_rows(
            responses=responses_rows,
            judgments=judgments_rows,
            run_started_at_utc=run_started_at_utc,
        ),
    )
    write_jsonl(output_dir / "trace.jsonl", trace_rows)

    summary = build_summary(responses_rows, judgments_rows)
    summary.update(
        {
            "run_id": run_id,
            "run_started_at_utc": run_started_at_utc,
            "selected_examples": len(examples),
            "datasets": dataset_stats,
            "judges": [judge_descriptor(j) for j in config.judges],
            "failed_items": failed_items,
            "num_failures": len(failed_items),
            "run_status": run_status,
            "interrupted_stage": interrupted_stage,
        }
    )
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_config.json",
        {
            "providers": sorted(list(config.providers.keys())),
            "candidates": [c.__dict__ for c in config.candidates],
            "judges": [j.__dict__ for j in config.judges],
            "datasets": [d.__dict__ for d in config.data.datasets],
            "retry": config.retry.__dict__,
            "cache": config.cache.__dict__,
            "run": config.run.__dict__,
        },
    )

    return summary
