from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ResponseRow(TypedDict):
    run_id: str
    run_started_at_utc: str
    dataset: str
    provenance: str
    judge_mode: str
    example_id: str
    candidate_name: str
    candidate_provider: str
    candidate_model: str
    candidate_settings: Dict[str, Any]
    request_id: str
    cache_key: Optional[str]
    cache_hit: bool
    response_source: str
    prompt_messages: List[Dict[str, str]]
    response_text: str
    usage: Dict[str, Any]
    latency_s: Optional[float]
    metadata: Dict[str, Any]
    reference_answer: Optional[str]


JudgmentRow = TypedDict(
    "JudgmentRow",
    {
        "run_id": str,
        "run_started_at_utc": str,
        "dataset": str,
        "provenance": str,
        "judge_mode": str,
        "example_id": str,
        "candidate_name": str,
        "judge_name": str,
        "judge_provider": str,
        "judge_model": str,
        "judge_settings": Dict[str, Any],
        "request_id": str,
        "cache_key": Optional[str],
        "cache_hit": bool,
        "score": float,
        "pass": bool,
        "rationale": str,
        "criteria": Dict[str, float],
        "parse_error": bool,
        "prbench_weighted_raw": Optional[float],
        "prbench_points_normalized": Optional[float],
        "prbench_points_clipped": Optional[float],
        "raw_judge": Dict[str, Any],
    },
)


class TraceRow(TypedDict, total=False):
    run_id: str
    event: str
    dataset: str
    example_id: str
    model_name: str
    provider: str
    request_id: str
    cache_key: Optional[str]
    cache_hit: bool
    request: Dict[str, Any]
    response: Dict[str, Any]
    criterion_id: str
    criterion_index: int


class NormalizedRow(TypedDict):
    example_id: str
    dataset: str
    provenance: str
    judge_mode: str
    instructions: str
    context: str
    messages: List[Dict[str, str]]
    reference_answer: Optional[str]
    metadata: Dict[str, Any]
