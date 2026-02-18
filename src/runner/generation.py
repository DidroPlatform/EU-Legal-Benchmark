from __future__ import annotations

import concurrent.futures
import sys
from typing import Any, Callable, Dict, List, Set, Tuple

from src.config import BenchmarkConfig, ModelConfig
from src.providers.base import BaseProvider
from src.types import LLMMessage, NormalizedExample


def _is_candidate_fatal_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "on-demand throughput isn’t supported" in message
        or "on-demand throughput isn't supported" in message
        or ("inference profile" in message and "bedrockexception" in message)
    )


def run_generation_phase(
    *,
    config: BenchmarkConfig,
    run_id: str,
    run_started_at_utc: str,
    examples: List[NormalizedExample],
    providers: Dict[str, BaseProvider],
    cache: Any,
    response_workers: int,
    response_rate_limiter: Any,
    provider_response_rate_limiters: Dict[str, Any],
    progress_mode: str,
    emit_progress: Callable[[str, str], None],
    progress_line: Callable[..., str],
    model_settings: Callable[[ModelConfig], Dict[str, Any]],
    to_jsonable_messages: Callable[[List[LLMMessage]], List[Dict[str, str]]],
    request_id: Callable[[str, str, str, str], str],
    build_request: Callable[[ModelConfig, List[LLMMessage], str], Any],
    run_model_call: Callable[..., Tuple[Dict[str, Any], bool, str]],
    build_candidate_messages: Callable[[NormalizedExample, str], List[LLMMessage]],
) -> Dict[str, Any]:
    total_items = len(config.candidates) * len(examples)

    generation_tasks: List[Tuple[int, ModelConfig, NormalizedExample]] = []
    display_index = 0
    for candidate in config.candidates:
        for example in examples:
            display_index += 1
            generation_tasks.append((display_index, candidate, example))

    emit_progress(
        progress_mode,
        progress_line(
            stage="response_phase_start",
            total_items=total_items,
            workers=min(response_workers, max(1, len(generation_tasks))),
            rpm=config.run.response_rate_limit_rpm,
        ),
    )

    def _generate_candidate_response(
        display_index: int, candidate: ModelConfig, example: NormalizedExample
    ) -> Dict[str, Any]:
        provider = providers[candidate.provider]
        candidate_messages = build_candidate_messages(example, config.run.default_system_prompt)
        response_req_id = request_id(run_id, candidate.name, example.id, stage="response")
        request = build_request(candidate, candidate_messages, response_req_id)

        def _wait_for_generation_slot(_: int) -> None:
            response_rate_limiter.wait()
            provider_limiter = provider_response_rate_limiters.get(candidate.provider)
            if provider_limiter is not None:
                provider_limiter.wait()

        emit_progress(
            progress_mode,
            progress_line(
                item=f"{display_index}/{total_items}",
                stage="response_started",
                candidate=candidate.name,
                dataset=example.dataset_name,
                example=example.id,
            ),
        )
        response_payload, cache_hit, cache_key = run_model_call(
            provider,
            request,
            cache,
            config.retry,
            stage="response",
            include_raw=config.run.include_raw_provider_response,
            before_attempt=_wait_for_generation_slot,
        )

        response_row = {
            "run_id": run_id,
            "run_started_at_utc": run_started_at_utc,
            "dataset": example.dataset_name,
            "provenance": example.provenance,
            "judge_mode": example.judge_mode,
            "example_id": example.id,
            "candidate_name": candidate.name,
            "candidate_provider": candidate.provider,
            "candidate_model": candidate.model,
            "candidate_settings": model_settings(candidate),
            "request_id": response_payload.get("request_id") or response_req_id,
            "cache_key": cache_key,
            "cache_hit": cache_hit,
            "prompt_messages": to_jsonable_messages(candidate_messages),
            "response_text": response_payload.get("text", ""),
            "usage": response_payload.get("usage", {}),
            "latency_s": response_payload.get("latency_s"),
            "metadata": example.metadata,
            "reference_answer": example.reference_answer,
        }

        generation_trace = {
            "run_id": run_id,
            "event": "generation_call",
            "dataset": example.dataset_name,
            "example_id": example.id,
            "model_name": candidate.name,
            "provider": candidate.provider,
            "request_id": response_row["request_id"],
            "cache_key": cache_key,
            "cache_hit": cache_hit,
            "request": {
                "messages": to_jsonable_messages(candidate_messages),
                "temperature": candidate.temperature,
                "top_p": candidate.top_p,
                "frequency_penalty": candidate.frequency_penalty,
                "presence_penalty": candidate.presence_penalty,
                "max_tokens": candidate.max_tokens,
                "seed": candidate.seed,
                "extra_body": candidate.extra_body,
            },
            "response": {
                "text": response_row["response_text"],
                "usage": response_row["usage"],
                "latency_s": response_row["latency_s"],
            },
        }

        return {
            "display_index": display_index,
            "candidate": candidate,
            "example": example,
            "response_row": response_row,
            "cache_hit": cache_hit,
            "generation_trace": generation_trace,
        }

    generation_results: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []
    interrupted = False
    generation_worker_count = min(response_workers, len(generation_tasks))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=generation_worker_count)
    inflight: Dict[concurrent.futures.Future, Tuple[int, ModelConfig, NormalizedExample]] = {}
    blocked_candidates: Set[str] = set()
    next_task_idx = 0

    def _mark_skipped_task(display_index: int, candidate: ModelConfig, example: NormalizedExample, reason: str) -> None:
        emit_progress(
            progress_mode,
            progress_line(
                item=f"{display_index}/{total_items}",
                stage="response_skipped",
                candidate=candidate.name,
                dataset=example.dataset_name,
                example=example.id,
                reason=reason,
            ),
        )
        failed_items.append(
            {
                "display_index": display_index,
                "candidate_name": candidate.name,
                "example_id": example.id,
                "dataset": example.dataset_name,
                "error": reason,
            }
        )

    def _submit_until_full() -> None:
        nonlocal next_task_idx
        while len(inflight) < generation_worker_count and next_task_idx < len(generation_tasks):
            display_index, candidate, example = generation_tasks[next_task_idx]
            next_task_idx += 1
            if candidate.name in blocked_candidates:
                _mark_skipped_task(
                    display_index,
                    candidate,
                    example,
                    "Skipped due to earlier fatal provider error for this candidate.",
                )
                continue

            emit_progress(
                progress_mode,
                progress_line(
                    item=f"{display_index}/{total_items}",
                    stage="response_queued",
                    candidate=candidate.name,
                    dataset=example.dataset_name,
                    example=example.id,
                ),
            )
            fut = pool.submit(_generate_candidate_response, display_index, candidate, example)
            inflight[fut] = (display_index, candidate, example)

    try:
        _submit_until_full()
        while inflight:
            done, _ = concurrent.futures.wait(
                set(inflight.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                di, cand, ex = inflight.pop(future)
                try:
                    generation_results.append(future.result())
                except Exception as exc:
                    print(
                        f"[error] generation failed for candidate={cand.name} "
                        f"dataset={ex.dataset_name} example={ex.id}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    failed_items.append(
                        {
                            "display_index": di,
                            "candidate_name": cand.name,
                            "example_id": ex.id,
                            "dataset": ex.dataset_name,
                            "error": str(exc),
                        }
                    )
                    if _is_candidate_fatal_error(exc) and cand.name not in blocked_candidates:
                        blocked_candidates.add(cand.name)
                        print(
                            f"[warning] disabling remaining tasks for candidate={cand.name} "
                            f"after fatal provider configuration error",
                            file=sys.stderr,
                            flush=True,
                        )
                _submit_until_full()
    except KeyboardInterrupt:
        interrupted = True
        emit_progress(
            progress_mode,
            progress_line(
                stage="response_phase_interrupted",
                total_items=total_items,
                completed_items=len(generation_results) + len(failed_items),
            ),
        )
        for future in inflight:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    else:
        pool.shutdown(wait=True, cancel_futures=False)

    generation_results.sort(key=lambda item: int(item["display_index"]))

    responses_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for generation_result in generation_results:
        display_index = int(generation_result["display_index"])
        candidate = generation_result["candidate"]
        example = generation_result["example"]
        response_row = generation_result["response_row"]
        cache_hit = bool(generation_result["cache_hit"])

        responses_rows.append(response_row)
        emit_progress(
            progress_mode,
            progress_line(
                item=f"{display_index}/{total_items}",
                stage="response_done",
                candidate=candidate.name,
                dataset=example.dataset_name,
                example=example.id,
                cache_hit=cache_hit,
                latency_s=response_row["latency_s"],
            ),
        )

        trace_rows.append(generation_result["generation_trace"])

    if not interrupted:
        emit_progress(progress_mode, progress_line(stage="response_phase_done", total_items=total_items))

    return {
        "generation_results": generation_results,
        "responses_rows": responses_rows,
        "trace_rows": trace_rows,
        "failed_items": failed_items,
        "interrupted": interrupted,
        "interrupted_stage": "generation" if interrupted else None,
    }
