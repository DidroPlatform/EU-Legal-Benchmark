from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Dict, List

from src.config import BenchmarkConfig, ModelConfig
from src.providers.base import BaseProvider
from src.types import JudgeResult, LLMMessage, NormalizedExample


def _build_judgment_row(
    *,
    run_id: str,
    run_started_at_utc: str,
    example: NormalizedExample,
    candidate_name: str,
    judge_name: str,
    judge_provider: str,
    judge_model: str,
    judge_settings: Dict[str, Any],
    request_id: str,
    cache_key: str | None,
    cache_hit: bool,
    result: JudgeResult,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "run_started_at_utc": run_started_at_utc,
        "dataset": example.dataset_name,
        "provenance": example.provenance,
        "judge_mode": example.judge_mode,
        "example_id": example.id,
        "candidate_name": candidate_name,
        "judge_name": judge_name,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_settings": judge_settings,
        "request_id": request_id,
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "score": result.score,
        "pass": result.passed,
        "rationale": result.rationale,
        "criteria": result.criteria,
        "parse_error": result.parse_error,
        "raw_judge": result.raw,
    }


def run_judge_phase(
    *,
    config: BenchmarkConfig,
    run_id: str,
    run_started_at_utc: str,
    total_items: int,
    generation_results: List[Dict[str, Any]],
    providers: Dict[str, BaseProvider],
    cache: Any,
    judge_workers: int,
    judge_rate_limiter: Any,
    progress_mode: str,
    emit_progress: Callable[[str, str], None],
    progress_line: Callable[..., str],
    request_id: Callable[[str, str, str, str], str],
    build_request: Callable[[ModelConfig, List[LLMMessage], str], Any],
    model_settings: Callable[[ModelConfig], Dict[str, Any]],
    judge_descriptor: Callable[[ModelConfig], Dict[str, Any]],
    to_jsonable_messages: Callable[[List[LLMMessage]], List[Dict[str, str]]],
    run_model_call: Callable[..., Any],
    grade_mcq_output: Callable[..., JudgeResult],
    build_judge_messages: Callable[..., List[LLMMessage]],
    build_rubric_criterion_judge_messages: Callable[..., List[LLMMessage]],
    parse_judge_output: Callable[..., JudgeResult],
    resolve_rubric_criterion_score: Callable[..., Any],
    apply_weighted_rubric_score: Callable[..., JudgeResult],
    google_provider_names: set[str],
) -> Dict[str, Any]:
    emit_progress(progress_mode, progress_line(stage="judge_phase_start", total_items=total_items))

    judgments_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for generation_result in generation_results:
        display_index = int(generation_result["display_index"])
        candidate = generation_result["candidate"]
        example = generation_result["example"]
        response_row = generation_result["response_row"]

        if example.judge_mode == "mcq":
            parsed = grade_mcq_output(
                example=example,
                candidate_text=response_row["response_text"],
                pass_threshold=config.run.judge_pass_threshold,
            )
            judge_req_id = request_id(
                run_id, "deterministic_mcq", f"{example.id}:{candidate.name}", stage="judge"
            )

            judgment_row = _build_judgment_row(
                run_id=run_id,
                run_started_at_utc=run_started_at_utc,
                example=example,
                candidate_name=candidate.name,
                judge_name="deterministic_mcq",
                judge_provider="programmatic",
                judge_model="exact_match_v1",
                judge_settings={},
                request_id=judge_req_id,
                cache_key=None,
                cache_hit=False,
                result=parsed,
            )
            judgments_rows.append(judgment_row)
            emit_progress(
                progress_mode,
                progress_line(
                    item=f"{display_index}/{total_items}",
                    stage="judge_done",
                    candidate=candidate.name,
                    dataset=example.dataset_name,
                    example=example.id,
                    judge_mode=example.judge_mode,
                    score=judgment_row["score"],
                    passed=judgment_row["pass"],
                    parse_error=judgment_row["parse_error"],
                ),
            )

            trace_rows.append(
                {
                    "run_id": run_id,
                    "event": "mcq_grade",
                    "dataset": example.dataset_name,
                    "example_id": example.id,
                    "model_name": "deterministic_mcq",
                    "provider": "programmatic",
                    "request_id": judgment_row["request_id"],
                    "cache_key": None,
                    "cache_hit": False,
                    "request": {
                        "expected_choice_ids": example.metadata.get("correct_choice_ids", []),
                    },
                    "response": {
                        "candidate_text": response_row["response_text"],
                        "parse_error": parsed.parse_error,
                        "score": parsed.score,
                        "criteria": parsed.criteria,
                    },
                }
            )
        else:
            if example.judge_mode == "rubric" and isinstance(example.rubric, list) and example.rubric:
                rubric_items = [item for item in example.rubric if isinstance(item, dict)]
                rubric_judges = config.judges or [config.judge]

                criterion_scores: Dict[str, float] = {}
                criterion_rationales: Dict[str, str] = {}
                criterion_call_details: List[Dict[str, Any]] = []
                any_parse_error = False
                any_cache_hit = False
                first_request_id = None

                def _run_rubric_criterion(idx: int, criterion: Dict[str, Any]) -> Dict[str, Any]:
                    criterion_id = str(criterion.get("id", f"criterion_{idx}")).strip()
                    judge_model = rubric_judges[(idx - 1) % len(rubric_judges)]
                    judge_provider = providers[judge_model.provider]
                    judge_req_id = request_id(
                        run_id,
                        judge_model.name,
                        f"{example.id}:{candidate.name}:{criterion_id}",
                        stage="judge",
                    )
                    judge_messages: List[LLMMessage] = []
                    judge_payload: Dict[str, Any] = {
                        "text": "",
                        "usage": {},
                        "latency_s": None,
                        "request_id": judge_req_id,
                    }
                    judge_cache_hit = False
                    judge_cache_key = None
                    error_message: str | None = None

                    try:
                        judge_messages = build_rubric_criterion_judge_messages(
                            example=example,
                            model_output=response_row["response_text"],
                            criterion=criterion,
                            criterion_index=idx,
                            pass_threshold=config.run.judge_pass_threshold,
                        )
                        judge_request = build_request(judge_model, judge_messages, judge_req_id)
                        if judge_rate_limiter is not None and judge_model.provider in google_provider_names:
                            judge_rate_limiter.wait()
                        judge_payload, judge_cache_hit, judge_cache_key = run_model_call(
                            judge_provider,
                            judge_request,
                            cache,
                            config.retry,
                            stage="judge",
                            include_raw=config.run.include_raw_provider_response,
                        )
                        parsed = parse_judge_output(
                            raw_text=judge_payload.get("text", "{}"),
                            fallback_pass_threshold=config.run.judge_pass_threshold,
                        )
                        criterion_score, _ = resolve_rubric_criterion_score(
                            criteria=parsed.criteria,
                            criterion=criterion,
                            criterion_index=idx,
                            fallback_score=parsed.score,
                        )
                    except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary
                        error_message = f"{type(exc).__name__}: {exc}"
                        judge_payload["error"] = error_message
                        parsed = JudgeResult(
                            score=0.0,
                            passed=False,
                            rationale=(
                                f"Judge call failed for criterion '{criterion_id}': {error_message}"
                            ),
                            criteria={},
                            raw={"error": error_message},
                            parse_error=True,
                        )
                        criterion_score = 0.0

                    return {
                        "criterion_id": criterion_id,
                        "criterion_index": idx,
                        "judge_model": judge_model,
                        "judge_messages": judge_messages,
                        "judge_req_id": judge_req_id,
                        "judge_payload": judge_payload,
                        "judge_cache_hit": judge_cache_hit,
                        "judge_cache_key": judge_cache_key,
                        "parsed": parsed,
                        "criterion_score": criterion_score,
                        "error": error_message,
                    }

                worker_count = min(judge_workers, len(rubric_items))
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
                    futures = [
                        pool.submit(_run_rubric_criterion, idx, criterion)
                        for idx, criterion in enumerate(rubric_items, start=1)
                    ]
                    criterion_results = [future.result() for future in concurrent.futures.as_completed(futures)]

                criterion_results.sort(key=lambda item: int(item["criterion_index"]))
                for result in criterion_results:
                    idx = int(result["criterion_index"])
                    criterion_id = str(result["criterion_id"])
                    judge_model = result["judge_model"]
                    judge_messages = result["judge_messages"]
                    judge_req_id = str(result["judge_req_id"])
                    judge_payload = result["judge_payload"]
                    judge_cache_hit = bool(result["judge_cache_hit"])
                    judge_cache_key = result["judge_cache_key"]
                    parsed = result["parsed"]
                    criterion_score = float(result["criterion_score"])
                    criterion_error = result.get("error")

                    criterion_scores[criterion_id] = criterion_score
                    criterion_rationales[criterion_id] = parsed.rationale
                    any_parse_error = any_parse_error or parsed.parse_error
                    any_cache_hit = any_cache_hit or judge_cache_hit
                    if first_request_id is None:
                        first_request_id = judge_payload.get("request_id") or judge_req_id

                    criterion_call = {
                        "criterion_id": criterion_id,
                        "criterion_index": idx,
                        "judge": judge_descriptor(judge_model),
                        "request_id": judge_payload.get("request_id") or judge_req_id,
                        "cache_key": judge_cache_key,
                        "cache_hit": judge_cache_hit,
                        "score": criterion_score,
                        "raw_score": parsed.score,
                        "rationale": parsed.rationale,
                        "parse_error": parsed.parse_error,
                        "raw": parsed.raw,
                        "error": criterion_error,
                    }
                    criterion_call_details.append(criterion_call)

                    trace_rows.append(
                        {
                            "run_id": run_id,
                            "event": "judge_call",
                            "dataset": example.dataset_name,
                            "example_id": example.id,
                            "criterion_id": criterion_id,
                            "criterion_index": idx,
                            "model_name": judge_model.name,
                            "provider": judge_model.provider,
                            "request_id": criterion_call["request_id"],
                            "cache_key": judge_cache_key,
                            "cache_hit": judge_cache_hit,
                            "request": {
                                "messages": to_jsonable_messages(judge_messages),
                                "temperature": judge_model.temperature,
                                "top_p": judge_model.top_p,
                                "frequency_penalty": judge_model.frequency_penalty,
                                "presence_penalty": judge_model.presence_penalty,
                                "max_tokens": judge_model.max_tokens,
                                "seed": judge_model.seed,
                                "extra_body": judge_model.extra_body,
                            },
                            "response": {
                                "text": judge_payload.get("text", ""),
                                "usage": judge_payload.get("usage", {}),
                                "latency_s": judge_payload.get("latency_s"),
                                "parse_error": parsed.parse_error,
                                "criterion_score": criterion_score,
                                "error": criterion_error,
                            },
                        }
                    )

                aggregate = JudgeResult(
                    score=0.0,
                    passed=False,
                    rationale="\n\n".join(
                        f"{criterion_id}: {text}" for criterion_id, text in criterion_rationales.items() if text
                    ),
                    criteria=criterion_scores,
                    raw={
                        "mode": "per_criterion_judges",
                        "assignment": "round_robin",
                        "calls": criterion_call_details,
                    },
                    parse_error=any_parse_error,
                )
                aggregate = apply_weighted_rubric_score(
                    parsed=aggregate,
                    example=example,
                    pass_threshold=config.run.judge_pass_threshold,
                )

                used_judges = {(j["judge"]["provider"], j["judge"]["model"]) for j in criterion_call_details}
                if len(used_judges) == 1 and criterion_call_details:
                    judge_provider_value = criterion_call_details[0]["judge"]["provider"]
                    judge_model_value = criterion_call_details[0]["judge"]["model"]
                else:
                    judge_provider_value = "mixed"
                    judge_model_value = "mixed"

                judgment_row = _build_judgment_row(
                    run_id=run_id,
                    run_started_at_utc=run_started_at_utc,
                    example=example,
                    candidate_name=candidate.name,
                    judge_name="rubric_multi_judge",
                    judge_provider=judge_provider_value,
                    judge_model=judge_model_value,
                    judge_settings={
                        "assignment": "round_robin",
                        "judges": [judge_descriptor(j) for j in rubric_judges],
                    },
                    request_id=first_request_id
                    or request_id(run_id, "rubric_multi_judge", f"{example.id}:{candidate.name}", "judge"),
                    cache_key=None,
                    cache_hit=any_cache_hit,
                    result=aggregate,
                )
                judgments_rows.append(judgment_row)
                emit_progress(
                    progress_mode,
                    progress_line(
                        item=f"{display_index}/{total_items}",
                        stage="judge_done",
                        candidate=candidate.name,
                        dataset=example.dataset_name,
                        example=example.id,
                        judge_mode=example.judge_mode,
                        score=judgment_row["score"],
                        passed=judgment_row["pass"],
                        parse_error=judgment_row["parse_error"],
                    ),
                )
            else:
                judge_model = config.judge
                judge_provider = providers[judge_model.provider]
                judge_messages = build_judge_messages(
                    example,
                    response_row["response_text"],
                    pass_threshold=config.run.judge_pass_threshold,
                )
                judge_req_id = request_id(run_id, judge_model.name, f"{example.id}:{candidate.name}", stage="judge")
                judge_request = build_request(judge_model, judge_messages, judge_req_id)
                if judge_rate_limiter is not None and judge_model.provider in google_provider_names:
                    judge_rate_limiter.wait()

                judge_cache_hit = False
                judge_cache_key = None
                judge_payload: Dict[str, Any] = {
                    "text": "",
                    "usage": {},
                    "latency_s": None,
                    "request_id": judge_req_id,
                }
                try:
                    judge_payload, judge_cache_hit, judge_cache_key = run_model_call(
                        judge_provider,
                        judge_request,
                        cache,
                        config.retry,
                        stage="judge",
                        include_raw=config.run.include_raw_provider_response,
                    )

                    parsed = parse_judge_output(
                        raw_text=judge_payload.get("text", "{}"),
                        fallback_pass_threshold=config.run.judge_pass_threshold,
                    )
                    parsed = apply_weighted_rubric_score(
                        parsed=parsed,
                        example=example,
                        pass_threshold=config.run.judge_pass_threshold,
                    )
                except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary
                    error_message = f"{type(exc).__name__}: {exc}"
                    judge_payload["error"] = error_message
                    parsed = JudgeResult(
                        score=0.0,
                        passed=False,
                        rationale=f"Judge call failed: {error_message}",
                        criteria={},
                        raw={"error": error_message},
                        parse_error=True,
                    )

                judgment_row = _build_judgment_row(
                    run_id=run_id,
                    run_started_at_utc=run_started_at_utc,
                    example=example,
                    candidate_name=candidate.name,
                    judge_name=judge_model.name,
                    judge_provider=judge_model.provider,
                    judge_model=judge_model.model,
                    judge_settings=model_settings(judge_model),
                    request_id=judge_payload.get("request_id") or judge_req_id,
                    cache_key=judge_cache_key,
                    cache_hit=judge_cache_hit,
                    result=parsed,
                )
                judgments_rows.append(judgment_row)
                emit_progress(
                    progress_mode,
                    progress_line(
                        item=f"{display_index}/{total_items}",
                        stage="judge_done",
                        candidate=candidate.name,
                        dataset=example.dataset_name,
                        example=example.id,
                        judge_mode=example.judge_mode,
                        score=judgment_row["score"],
                        passed=judgment_row["pass"],
                        parse_error=judgment_row["parse_error"],
                    ),
                )

                trace_rows.append(
                    {
                        "run_id": run_id,
                        "event": "judge_call",
                        "dataset": example.dataset_name,
                        "example_id": example.id,
                        "model_name": judge_model.name,
                        "provider": judge_model.provider,
                        "request_id": judgment_row["request_id"],
                        "cache_key": judge_cache_key,
                        "cache_hit": judge_cache_hit,
                        "request": {
                            "messages": to_jsonable_messages(judge_messages),
                            "temperature": judge_model.temperature,
                            "top_p": judge_model.top_p,
                            "frequency_penalty": judge_model.frequency_penalty,
                            "presence_penalty": judge_model.presence_penalty,
                            "max_tokens": judge_model.max_tokens,
                            "seed": judge_model.seed,
                            "extra_body": judge_model.extra_body,
                        },
                        "response": {
                            "text": judge_payload.get("text", ""),
                            "usage": judge_payload.get("usage", {}),
                            "latency_s": judge_payload.get("latency_s"),
                            "parse_error": parsed.parse_error,
                            "error": judge_payload.get("error"),
                        },
                    }
                )

    emit_progress(progress_mode, progress_line(stage="judge_phase_done", total_items=total_items))

    return {
        "judgments_rows": judgments_rows,
        "trace_rows": trace_rows,
    }
