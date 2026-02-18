from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

if importlib.util.find_spec("src") is not None:
    from src.cache import DiskCache
    from src.config import BenchmarkConfig, ModelConfig, RetryConfig
    from src.data.loader import load_examples
    from src.data.schema import validate_jsonl_file
    from src.judge.judge import (
        apply_weighted_rubric_score,
        build_judge_messages,
        build_rubric_criterion_judge_messages,
        parse_judge_output,
        resolve_rubric_criterion_score,
    )
    from src.judge.mcq import grade_mcq_output
    from src.prompting.templates import build_candidate_messages
    from src.providers import build_provider
    from src.providers.base import BaseProvider
    from src.retry import with_retries
    from src.runner import run as _runner_run
    from src.runner.helpers import (
        _build_request,
        _cache_key_payload,
        _emit_progress,
        _judge_descriptor,
        _model_settings,
        _progress_enabled,
        _progress_line,
        _request_id,
        _to_jsonable_messages,
    )
    from src.runner.output import _merge_scored_rows, _summary, _write_jsonl
    from src.runner.rate_limiter import PerMinuteRateLimiter
    from src.setup_checks import check_setup, required_provider_names
    from src.types import LLMRequest, NormalizedExample
else:
    raise ModuleNotFoundError(
        "Could not resolve project imports. Run from repo root so `src` is importable."
    )


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _run_model_call(
    provider: BaseProvider,
    request: LLMRequest,
    cache: DiskCache,
    retry_cfg: RetryConfig,
    stage: str,
    include_raw: bool,
    before_attempt: Any | None = None,
) -> Tuple[Dict[str, Any], bool, str]:
    key = cache.make_key(_cache_key_payload(request, stage=stage))
    cached = cache.get(key)
    if cached is not None:
        return cached, True, key

    response = with_retries(
        lambda: provider.generate(request, include_raw=include_raw),
        retry_cfg,
        before_attempt=before_attempt,
    )
    payload = {
        "provider": response.provider,
        "model": response.model,
        "text": response.text,
        "usage": response.usage,
        "latency_s": response.latency_s,
        "request_id": response.request_id,
        "raw_response": response.raw_response,
    }
    cache.set(key, payload)
    return payload, False, key


def _load_all_examples(config: BenchmarkConfig) -> Tuple[List[NormalizedExample], List[Dict[str, Any]]]:
    examples: List[NormalizedExample] = []
    dataset_stats: List[Dict[str, Any]] = []

    for dataset in config.data.datasets:
        if not dataset.enabled:
            continue
        rows = load_examples(dataset)
        examples.extend(rows)
        dataset_stats.append(
            {
                "dataset": dataset.name,
                "path": dataset.path,
                "provenance": dataset.provenance,
                "judge_mode": dataset.judge_mode,
                "selected_examples": len(rows),
            }
        )

    return examples, dataset_stats


def _validate_canonical_inputs(config: BenchmarkConfig) -> None:
    problems: List[str] = []

    for dataset in config.data.datasets:
        if not dataset.enabled:
            continue
        result = validate_jsonl_file(dataset.path)
        invalid = int(result.get("invalid_rows", 0))
        if invalid <= 0:
            continue

        rows = int(result.get("rows", 0))
        details = []
        for err in result.get("errors", [])[:5]:
            line = err.get("line")
            row_id = err.get("id")
            msg = "; ".join(err.get("errors", []))
            details.append(f"line={line}, id={row_id}: {msg}")
        detail_text = "\n      ".join(details) if details else "(no details)"

        problems.append(
            f"- dataset='{dataset.name}' path='{dataset.path}' invalid_rows={invalid}/{rows}\n"
            f"      {detail_text}"
        )

    if problems:
        raise ValueError(
            "Canonical input validation failed for legal_eval_v1.\n"
            "Fix the dataset files before running the benchmark.\n"
            + "\n".join(problems)
        )


def run(config: BenchmarkConfig, limit_override: int | None = None, progress_mode: str = "log") -> Path:
    return _runner_run(
        config,
        limit_override=limit_override,
        progress_mode=progress_mode,
        validate_canonical_inputs=_validate_canonical_inputs,
        load_all_examples=_load_all_examples,
        required_provider_names=required_provider_names,
        build_provider=build_provider,
        run_model_call=_run_model_call,
        grade_mcq_output=grade_mcq_output,
        build_candidate_messages=build_candidate_messages,
        build_judge_messages=build_judge_messages,
        build_rubric_criterion_judge_messages=build_rubric_criterion_judge_messages,
        parse_judge_output=parse_judge_output,
        resolve_rubric_criterion_score=resolve_rubric_criterion_score,
        apply_weighted_rubric_score=apply_weighted_rubric_score,
        per_minute_rate_limiter_cls=PerMinuteRateLimiter,
        emit_progress=_emit_progress,
        progress_line=_progress_line,
        request_id=_request_id,
        to_jsonable_messages=_to_jsonable_messages,
        build_request=_build_request,
        model_settings=_model_settings,
        judge_descriptor=_judge_descriptor,
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark scaffold end-to-end.")
    parser.add_argument("--config", default="config.example.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional global limit across all datasets.")
    parser.add_argument(
        "--progress",
        choices=["log", "off"],
        default="log",
        help="Progress output mode. 'log' prints per-item status lines; 'off' disables progress output.",
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Validate environment and dependencies for configured candidate/judge providers, then exit.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    _load_dotenv_if_available()
    config = BenchmarkConfig.from_yaml(args.config)

    report = check_setup(config)
    if report.warnings:
        for warning in report.warnings:
            print(f"[setup warning] {warning}")
    if report.errors:
        for error in report.errors:
            print(f"[setup error] {error}")
        raise SystemExit(2)
    if args.check_setup:
        print("Setup check passed.")
        return

    out_dir = run(config, limit_override=args.limit, progress_mode=args.progress)
    summary_path = out_dir / "summary.json"
    run_status = "completed"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        run_status = str(summary.get("run_status") or "completed")
    if run_status == "interrupted":
        print(f"Run interrupted. Partial artifacts written to: {out_dir}")
    else:
        print(f"Run completed. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
