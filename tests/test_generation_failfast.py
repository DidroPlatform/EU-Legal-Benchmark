from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import run as run_module
from src.config import BenchmarkConfig, CacheConfig, DataConfig, DatasetConfig, ModelConfig, ProviderConfig, RunConfig
from src.types import JudgeResult, LLMMessage, NormalizedExample


class TestGenerationFailFast(unittest.TestCase):
    def _build_config(self, root: Path) -> BenchmarkConfig:
        dataset_path = root / "dataset.jsonl"
        dataset_path.write_text(
            "\n".join(
                [
                    '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"Q1","reference_answers":["A1"]}',
                    '{"schema_version":"legal_eval_v1","id":"ex2","dataset":"d","task_type":"reference_qa","prompt":"Q2","reference_answers":["A2"]}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        candidates = [
            ModelConfig(name="bedrock_cand", provider="openai", model="bedrock/anthropic.claude-opus-4-6-v1"),
            ModelConfig(name="healthy_cand", provider="openai", model="openai/gpt-4o-mini"),
        ]
        judge = ModelConfig(name="judge", provider="openai", model="openai/gpt-4o-mini")
        return BenchmarkConfig(
            providers={"openai": ProviderConfig(api_key_env=None)},
            candidates=candidates,
            judge=judge,
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
            cache=CacheConfig(enabled=False, dir=str(root / "cache")),
            run=RunConfig(output_dir=str(root / "outputs"), response_parallel_workers=1),
        )

    @staticmethod
    def _examples() -> list[NormalizedExample]:
        return [
            NormalizedExample(
                id="ex1",
                dataset_name="d",
                provenance="canonical:test",
                judge_mode="reference",
                instructions="Answer.",
            ),
            NormalizedExample(
                id="ex2",
                dataset_name="d",
                provenance="canonical:test",
                judge_mode="reference",
                instructions="Answer.",
            ),
        ]

    @staticmethod
    def _fake_parse_judge_output(raw_text: str, fallback_pass_threshold: float) -> JudgeResult:
        return JudgeResult(
            score=1.0,
            passed=True,
            rationale="ok",
            criteria={},
            raw={},
            parse_error=False,
        )

    def test_fatal_candidate_error_skips_remaining_candidate_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            output = StringIO()
            response_calls = {"bedrock": 0, "healthy": 0}

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                if stage == "judge":
                    return (
                        {
                            "provider": request.provider,
                            "model": request.model,
                            "text": '{"score": 1.0, "pass": true}',
                            "usage": {},
                            "latency_s": 0.01,
                            "request_id": "judge-request-id",
                            "raw_response": None,
                        },
                        False,
                        "judge-cache-key",
                    )

                if request.model.startswith("bedrock/"):
                    response_calls["bedrock"] += 1
                    raise Exception(
                        "litellm.BadRequestError: BedrockException - {\"message\":\"Invocation of model ID anthropic.claude-opus-4-6-v1 with on-demand throughput isn’t supported. Retry your request with the ID or ARN of an inference profile that contains this model.\"}"
                    )

                response_calls["healthy"] += 1
                return (
                    {
                        "provider": request.provider,
                        "model": request.model,
                        "text": "candidate answer",
                        "usage": {},
                        "latency_s": 0.01,
                        "request_id": "response-request-id",
                        "raw_response": None,
                    },
                    False,
                    "response-cache-key",
                )

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=(
                        self._examples(),
                        [{"dataset": "d", "path": config.data.datasets[0].path, "selected_examples": 2}],
                    ),
                ),
                mock.patch.object(run_module, "required_provider_names", return_value={"openai"}),
                mock.patch.object(run_module, "build_provider", return_value=object()),
                mock.patch.object(
                    run_module,
                    "build_candidate_messages",
                    return_value=[LLMMessage(role="user", content="prompt")],
                ),
                mock.patch.object(
                    run_module,
                    "build_judge_messages",
                    return_value=[LLMMessage(role="user", content="judge prompt")],
                ),
                mock.patch.object(run_module, "_run_model_call", side_effect=fake_model_call),
                mock.patch.object(run_module, "parse_judge_output", side_effect=self._fake_parse_judge_output),
                mock.patch.object(run_module, "apply_weighted_rubric_score", side_effect=lambda parsed, **_: parsed),
                redirect_stdout(output),
            ):
                out_dir = run_module.run(config=config, progress_mode="log")

            self.assertEqual(response_calls["bedrock"], 1)
            self.assertEqual(response_calls["healthy"], 2)

            with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("num_failures"), 2)
            errors = [item.get("error", "") for item in summary.get("failed_items", [])]
            self.assertTrue(any("Skipped due to earlier fatal provider error" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
