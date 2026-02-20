from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.config import BenchmarkConfig
from src.judge.judge import (
    apply_policy_score_postprocessing,
    apply_weighted_rubric_score,
    build_judge_messages,
    build_rubric_criterion_judge_messages,
    parse_judge_output,
    resolve_rubric_criterion_score,
)
from src.judge.mcq import grade_mcq_output
from src.prompting.templates import build_candidate_messages
from src.providers import build_provider
from src.runner import run as _runner_run
from src.runner.bootstrap import (
    build_runner_services as _build_runner_services_impl,
    load_all_examples as _load_all_examples_impl,
    run_model_call as _run_model_call_impl,
    validate_canonical_inputs as _validate_canonical_inputs_impl,
)
from src.runner.helpers import (
    _cache_key_payload,
    _progress_enabled,
    _progress_line,
)
from src.runner.rate_limiter import PerMinuteRateLimiter
from src.runtime.bootstrap import load_dotenv_if_available
from src.setup_checks import check_setup, required_provider_names


# Keep thin compatibility aliases so tests can patch run-level seams while
# the implementations live in src/runner/bootstrap.py.
_run_model_call = _run_model_call_impl
_load_all_examples = _load_all_examples_impl
_validate_canonical_inputs = _validate_canonical_inputs_impl


def _build_runner_services():
    return _build_runner_services_impl(
        validate_canonical_inputs_fn=_validate_canonical_inputs,
        load_all_examples_fn=_load_all_examples,
        required_provider_names_fn=required_provider_names,
        build_provider_fn=build_provider,
        run_model_call_fn=_run_model_call,
        build_candidate_messages_fn=build_candidate_messages,
        grade_mcq_output_fn=grade_mcq_output,
        build_judge_messages_fn=build_judge_messages,
        build_rubric_criterion_judge_messages_fn=build_rubric_criterion_judge_messages,
        parse_judge_output_fn=parse_judge_output,
        resolve_rubric_criterion_score_fn=resolve_rubric_criterion_score,
        apply_weighted_rubric_score_fn=apply_weighted_rubric_score,
        apply_policy_score_postprocessing_fn=apply_policy_score_postprocessing,
    )


def run(config: BenchmarkConfig, limit_override: int | None = None, progress_mode: str = "log") -> Path:
    services = _build_runner_services()
    return _runner_run(
        config,
        limit_override=limit_override,
        progress_mode=progress_mode,
        services=services,
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
    load_dotenv_if_available()
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
