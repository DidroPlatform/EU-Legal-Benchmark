from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import (
    BenchmarkConfig,
    DataConfig,
    DatasetConfig,
    ModelConfig,
    ProviderConfig,
    RetryConfig,
    RunConfig,
)
from src.runner.generation import (
    _plan_generation_tasks,
    _validate_preloaded_responses_coverage,
)
from src.runner.judging import _dispatch_judge_strategy, _run_judge_call
from src.types import JudgeResult, LLMMessage, NormalizedExample


class TestRunnerRefactorHelpers(unittest.TestCase):
    def _config(self, dataset_path: str) -> BenchmarkConfig:
        candidate_a = ModelConfig(name="cand_a", provider="openai", model="openai/gpt-4o-mini")
        candidate_b = ModelConfig(name="cand_b", provider="openai", model="openai/gpt-4o-mini")
        judge = ModelConfig(name="judge", provider="openai", model="openai/gpt-4o-mini")
        return BenchmarkConfig(
            providers={"openai": ProviderConfig(api_key_env=None)},
            candidates=[candidate_a, candidate_b],
            judges=[judge],
            data=DataConfig(
                datasets=[
                    DatasetConfig(
                        name="d",
                        path=dataset_path,
                        provenance="canonical:test",
                        judge_mode="reference",
                    )
                ]
            ),
            retry=RetryConfig(max_attempts=1),
            run=RunConfig(output_dir="outputs"),
        )

    @staticmethod
    def _examples() -> list[NormalizedExample]:
        return [
            NormalizedExample(
                id="ex1",
                dataset_name="d",
                provenance="canonical:test",
                judge_mode="reference",
                instructions="Q1",
            ),
            NormalizedExample(
                id="ex2",
                dataset_name="d",
                provenance="canonical:test",
                judge_mode="reference",
                instructions="Q2",
            ),
        ]

    def test_plan_generation_tasks_is_candidate_major_with_stable_display_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset_path = str(Path(td) / "dataset.jsonl")
            Path(dataset_path).write_text("", encoding="utf-8")
            config = self._config(dataset_path)

            tasks, total_items = _plan_generation_tasks(config, self._examples())

            self.assertEqual(total_items, 4)
            self.assertEqual(
                [(idx, candidate.name, example.id) for idx, candidate, example in tasks],
                [
                    (1, "cand_a", "ex1"),
                    (2, "cand_a", "ex2"),
                    (3, "cand_b", "ex1"),
                    (4, "cand_b", "ex2"),
                ],
            )

    def test_validate_preloaded_responses_coverage_reports_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset_path = str(Path(td) / "dataset.jsonl")
            Path(dataset_path).write_text("", encoding="utf-8")
            config = self._config(dataset_path)
            tasks, _ = _plan_generation_tasks(config, self._examples())

            with self.assertRaises(ValueError) as ctx:
                _validate_preloaded_responses_coverage(
                    "prefilled",
                    tasks,
                    {
                        ("ex1", "cand_a"): "a1",
                        ("ex2", "cand_a"): "a2",
                    },
                )

            self.assertIn("Missing prefilled responses for selected tasks", str(ctx.exception))
            self.assertIn("ex1:cand_b", str(ctx.exception))

    def test_dispatch_judge_strategy_routes_by_mode(self) -> None:
        mcq = NormalizedExample(
            id="mcq",
            dataset_name="d",
            provenance="canonical:test",
            judge_mode="mcq",
            instructions="Q",
        )
        rubric = NormalizedExample(
            id="rubric",
            dataset_name="d",
            provenance="canonical:test",
            judge_mode="rubric",
            instructions="Q",
            rubric=[{"id": "c1", "title": "Accuracy"}],
        )
        reference = NormalizedExample(
            id="ref",
            dataset_name="d",
            provenance="canonical:test",
            judge_mode="reference",
            instructions="Q",
        )

        self.assertEqual(_dispatch_judge_strategy(mcq), "mcq")
        self.assertEqual(_dispatch_judge_strategy(rubric), "rubric_multi_judge")
        self.assertEqual(_dispatch_judge_strategy(reference), "single_judge")

    def test_run_judge_call_error_path_returns_parse_error_compatible_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset_path = str(Path(td) / "dataset.jsonl")
            Path(dataset_path).write_text("", encoding="utf-8")
            config = self._config(dataset_path)
            judge_model = config.primary_judge

            result = _run_judge_call(
                judge_provider=object(),
                judge_model=judge_model,
                judge_req_id="req-1",
                judge_messages=[LLMMessage(role="user", content="grade")],
                build_request=lambda model, messages, req_id: {
                    "model": model.model,
                    "request_id": req_id,
                    "messages": messages,
                },
                run_model_call=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
                parse_judge_output=lambda **kwargs: JudgeResult(
                    score=1.0,
                    passed=True,
                    rationale="ok",
                    criteria={"overall": 1.0},
                    raw={},
                    parse_error=False,
                ),
                cache=None,
                config=config,
                judge_rate_limiter=None,
                google_provider_names=set(),
                error_result_builder=lambda error: JudgeResult(
                    score=0.0,
                    passed=False,
                    rationale=error,
                    criteria={},
                    raw={"error": error},
                    parse_error=True,
                ),
            )

            self.assertTrue(result.parsed.parse_error)
            self.assertEqual(result.judge_cache_hit, False)
            self.assertIsNone(result.judge_cache_key)
            self.assertIn("RuntimeError: boom", result.judge_payload.get("error", ""))
