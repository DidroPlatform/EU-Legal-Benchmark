from __future__ import annotations

import datetime as dt
import secrets
from pathlib import Path
from typing import Any, Dict, List

from src.cache import DiskCache
from src.config import BenchmarkConfig
from src.runner.row_types import NormalizedRow
from src.runner.services import RunnerServices
from src.types import GOOGLE_PROVIDER_NAMES

from .context import RunnerContext
from .generation import run_generation_phase
from .judging import run_judge_phase
from .output import write_run_outputs


def _recompute_dataset_stats_after_limit(
    dataset_stats: List[Dict[str, Any]],
    examples: List[Any],
) -> List[Dict[str, Any]]:
    selected_by_dataset: Dict[str, int] = {}
    for example in examples:
        dataset_name = str(example.dataset_name)
        selected_by_dataset[dataset_name] = selected_by_dataset.get(dataset_name, 0) + 1

    updated_stats: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for stat in dataset_stats:
        dataset_name = str(stat.get("dataset", ""))
        row = dict(stat)
        row["selected_examples"] = selected_by_dataset.get(dataset_name, 0)
        updated_stats.append(row)
        seen.add(dataset_name)

    for dataset_name, selected in selected_by_dataset.items():
        if dataset_name in seen:
            continue
        updated_stats.append({"dataset": dataset_name, "selected_examples": selected})

    return updated_stats


def run(
    config: BenchmarkConfig,
    limit_override: int | None = None,
    progress_mode: str = "log",
    *,
    services: RunnerServices,
) -> Path:
    config.validate()
    run_started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    run_id = config.run.run_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
    )
    services.bootstrap.validate_canonical_inputs(config)

    examples, dataset_stats = services.bootstrap.load_all_examples(config)
    if limit_override is not None:
        examples = examples[: max(0, limit_override)]
    dataset_stats = _recompute_dataset_stats_after_limit(dataset_stats, examples)

    if not examples:
        raise ValueError("No examples selected after dataset filtering.")

    run_dir = Path(config.run.runs_root) / run_id
    output_dir = run_dir / Path(config.run.output_dir).name
    cache_dir = run_dir / Path(config.cache.dir).name
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = DiskCache(str(cache_dir), enabled=config.cache.enabled)
    required_providers = services.bootstrap.required_provider_names(config)
    providers = {
        name: services.bootstrap.build_provider(name, config)
        for name in required_providers
    }
    judge_uses_google = any(j.provider in GOOGLE_PROVIDER_NAMES for j in config.judges)
    response_workers = max(1, int(config.run.response_parallel_workers))
    response_rate_limiter = services.infrastructure.per_minute_rate_limiter_cls(
        config.run.response_rate_limit_rpm
    )
    provider_response_rate_limiters = {
        provider_name: services.infrastructure.per_minute_rate_limiter_cls(rpm)
        for provider_name, rpm in config.run.provider_response_rate_limit_rpm.items()
    }
    judge_workers = max(1, int(config.run.judge_parallel_workers))
    judge_rate_limiter = (
        services.infrastructure.per_minute_rate_limiter_cls(config.run.judge_rate_limit_rpm)
        if judge_uses_google and config.run.judge_rate_limit_rpm > 0
        else None
    )
    total_items = len(config.candidates) * len(examples)

    ctx = RunnerContext(
        config=config,
        run_id=run_id,
        run_started_at_utc=run_started_at_utc,
        providers=providers,
        cache=cache,
        progress_mode=progress_mode,
        google_provider_names=frozenset(GOOGLE_PROVIDER_NAMES),
        emit_progress=services.infrastructure.emit_progress,
        progress_line=services.infrastructure.progress_line,
        model_settings=services.infrastructure.model_settings,
        to_jsonable_messages=services.infrastructure.to_jsonable_messages,
        request_id=services.infrastructure.request_id,
        build_request=services.infrastructure.build_request,
        run_model_call=services.generation.run_model_call,
        judge_descriptor=services.infrastructure.judge_descriptor,
    )

    try:
        services.infrastructure.emit_progress(
            progress_mode,
            services.infrastructure.progress_line(
                stage="start",
                run_id=run_id,
                examples=len(examples),
                candidates=len(config.candidates),
                total_items=total_items,
            ),
        )

        normalized_rows: List[NormalizedRow] = []
        for example in examples:
            normalized_rows.append(
                {
                    "example_id": example.id,
                    "dataset": example.dataset_name,
                    "provenance": example.provenance,
                    "judge_mode": example.judge_mode,
                    "instructions": example.instructions,
                    "context": example.context,
                    "messages": services.infrastructure.to_jsonable_messages(example.messages),
                    "reference_answer": example.reference_answer,
                    "metadata": example.metadata,
                }
            )

        generation = run_generation_phase(
            ctx=ctx,
            examples=examples,
            response_workers=response_workers,
            response_rate_limiter=response_rate_limiter,
            provider_response_rate_limiters=provider_response_rate_limiters,
            build_candidate_messages=services.generation.build_candidate_messages,
        )

        judging_rows = []
        judging_trace_rows = []
        judging_failed_items = []
        judging_interrupted = False
        judging_interrupted_stage: str | None = None
        if not generation.interrupted:
            judging = run_judge_phase(
                ctx=ctx,
                total_items=total_items,
                generation_results=generation.generation_results,
                judge_workers=judge_workers,
                judge_rate_limiter=judge_rate_limiter,
                judging_services=services.judging,
            )
            judging_rows = judging.judgments_rows
            judging_trace_rows = judging.trace_rows
            judging_failed_items = judging.failed_items
            judging_interrupted = judging.interrupted
            judging_interrupted_stage = judging.interrupted_stage

        run_interrupted = generation.interrupted or judging_interrupted
        interrupted_stage = generation.interrupted_stage or judging_interrupted_stage

        all_trace_rows = [*generation.trace_rows, *judging_trace_rows]
        write_run_outputs(
            output_dir=output_dir,
            config=config,
            run_id=run_id,
            run_started_at_utc=run_started_at_utc,
            examples=examples,
            dataset_stats=dataset_stats,
            normalized_rows=normalized_rows,
            responses_rows=generation.responses_rows,
            judgments_rows=judging_rows,
            trace_rows=all_trace_rows,
            failed_items=[*generation.failed_items, *judging_failed_items],
            run_status="interrupted" if run_interrupted else "completed",
            interrupted_stage=interrupted_stage,
            judge_descriptor=services.infrastructure.judge_descriptor,
        )

        return output_dir
    finally:
        for provider in providers.values():
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                close_fn()
