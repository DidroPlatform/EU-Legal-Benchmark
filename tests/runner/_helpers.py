from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from unittest import mock

import run as run_module
from src.config import BenchmarkConfig, CacheConfig, DataConfig, DatasetConfig, ModelConfig, ProviderConfig, RunConfig
from src.types import JudgeResult, LLMMessage, NormalizedExample


_NO_PATCH = object()


def _candidate_name(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if index < len(alphabet):
        return f"cand_{alphabet[index]}"
    return f"cand_{index + 1}"


def build_runner_config(
    root: Path,
    *,
    candidate_names: Sequence[str] | None = None,
    candidate_count: int = 1,
    response_parallel_workers: int = 4,
    response_rate_limit_rpm: int = 50,
    judge_parallel_workers: int | None = None,
    cache_enabled: bool = False,
) -> BenchmarkConfig:
    dataset_path = root / "dataset.jsonl"
    dataset_path.write_text(
        '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"Q","reference_answers":["A"]}\n',
        encoding="utf-8",
    )

    names = list(candidate_names or [])
    if not names:
        if candidate_count <= 1:
            names = ["cand"]
        else:
            names = [_candidate_name(i) for i in range(candidate_count)]

    candidates = [ModelConfig(name=name, provider="openai", model="openai/gpt-4o-mini") for name in names]
    judge = ModelConfig(name="judge", provider="openai", model="openai/gpt-4o-mini")
    run_cfg = RunConfig(
        output_dir=str(root / "outputs"),
        response_parallel_workers=response_parallel_workers,
        response_rate_limit_rpm=response_rate_limit_rpm,
    )
    if judge_parallel_workers is not None:
        run_cfg.judge_parallel_workers = judge_parallel_workers

    return BenchmarkConfig(
        providers={"openai": ProviderConfig(api_key_env=None)},
        candidates=candidates,
        judges=[judge],
        data=DataConfig(
            datasets=[
                DatasetConfig(
                    name="d",
                    path=str(dataset_path),
                    provenance="canonical:test",
                    judge_mode="reference",
                )
            ]
        ),
        cache=CacheConfig(enabled=cache_enabled, dir=str(root / "cache")),
        run=run_cfg,
    )


def make_example(
    *,
    example_id: str = "ex1",
    instructions: str = "Answer",
    judge_mode: str = "reference",
    dataset_name: str = "d",
    provenance: str = "canonical:test",
    rubric: list[dict[str, Any]] | None = None,
) -> NormalizedExample:
    kwargs: dict[str, Any] = {}
    if rubric is not None:
        kwargs["rubric"] = rubric
    return NormalizedExample(
        id=example_id,
        dataset_name=dataset_name,
        provenance=provenance,
        judge_mode=judge_mode,
        instructions=instructions,
        **kwargs,
    )


def make_examples(
    count: int,
    *,
    instructions: str = "Answer",
    judge_mode: str = "reference",
    dataset_name: str = "d",
    provenance: str = "canonical:test",
) -> list[NormalizedExample]:
    return [
        make_example(
            example_id=f"ex{i + 1}",
            instructions=instructions,
            judge_mode=judge_mode,
            dataset_name=dataset_name,
            provenance=provenance,
        )
        for i in range(count)
    ]


def dataset_stats_for(config: BenchmarkConfig, selected_examples: int) -> list[dict[str, Any]]:
    return [
        {
            "dataset": "d",
            "path": config.data.datasets[0].path,
            "selected_examples": selected_examples,
        }
    ]


def fake_ok_judge_result(raw_text: str, fallback_pass_threshold: float) -> JudgeResult:
    del raw_text, fallback_pass_threshold
    return JudgeResult(
        score=1.0,
        passed=True,
        rationale="ok",
        criteria={},
        raw={},
        parse_error=False,
    )


def _identity_weighted(parsed: JudgeResult, **kwargs: Any) -> JudgeResult:
    del kwargs
    return parsed


@contextmanager
def patched_runner_env(
    *,
    config: BenchmarkConfig,
    examples: Sequence[NormalizedExample],
    run_model_call: Callable[..., tuple[dict[str, Any], bool, str]],
    parse_judge_output: Callable[[str, float], JudgeResult] | None = fake_ok_judge_result,
    apply_weighted_rubric_score: Callable[..., JudgeResult] | None = _identity_weighted,
    candidate_messages: list[LLMMessage] | None = None,
    judge_messages: list[LLMMessage] | None = None,
    rubric_judge_messages: list[LLMMessage] | None = None,
    limiter_wait_return: Any = _NO_PATCH,
) -> Iterator[dict[str, mock.MagicMock]]:
    if candidate_messages is None:
        candidate_messages = [LLMMessage(role="user", content="prompt")]
    if judge_messages is None:
        judge_messages = [LLMMessage(role="user", content="judge prompt")]
    if rubric_judge_messages is None:
        rubric_judge_messages = [LLMMessage(role="user", content="judge prompt")]

    handles: dict[str, mock.MagicMock] = {}
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None))
        stack.enter_context(
            mock.patch.object(
                run_module,
                "_load_all_examples",
                return_value=(list(examples), dataset_stats_for(config, selected_examples=len(examples))),
            )
        )
        stack.enter_context(mock.patch.object(run_module, "required_provider_names", return_value={"openai"}))
        stack.enter_context(mock.patch.object(run_module, "build_provider", return_value=object()))
        stack.enter_context(
            mock.patch.object(
                run_module,
                "build_candidate_messages",
                return_value=list(candidate_messages),
            )
        )
        stack.enter_context(
            mock.patch.object(
                run_module,
                "build_judge_messages",
                return_value=list(judge_messages),
            )
        )
        stack.enter_context(
            mock.patch.object(
                run_module,
                "build_rubric_criterion_judge_messages",
                return_value=list(rubric_judge_messages),
            )
        )
        stack.enter_context(mock.patch.object(run_module, "_run_model_call", side_effect=run_model_call))

        if parse_judge_output is not None:
            stack.enter_context(mock.patch.object(run_module, "parse_judge_output", side_effect=parse_judge_output))
        if apply_weighted_rubric_score is not None:
            stack.enter_context(
                mock.patch.object(
                    run_module,
                    "apply_weighted_rubric_score",
                    side_effect=apply_weighted_rubric_score,
                )
            )
        if limiter_wait_return is not _NO_PATCH:
            handles["wait"] = stack.enter_context(
                mock.patch.object(run_module.PerMinuteRateLimiter, "wait", return_value=limiter_wait_return)
            )

        yield handles
