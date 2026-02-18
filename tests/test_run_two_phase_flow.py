from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run as run_module
from src.config import BenchmarkConfig, CacheConfig, DataConfig, DatasetConfig, ModelConfig, ProviderConfig, RunConfig
from src.types import JudgeResult, LLMMessage, NormalizedExample


class TestRunTwoPhaseFlow(unittest.TestCase):
    def _build_config(self, root: Path) -> BenchmarkConfig:
        dataset_path = root / "dataset.jsonl"
        dataset_path.write_text(
            '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"Q","reference_answers":["A"]}\n',
            encoding="utf-8",
        )
        candidates = [
            ModelConfig(name="cand_a", provider="openai", model="openai/gpt-4o-mini"),
            ModelConfig(name="cand_b", provider="openai", model="openai/gpt-4o-mini"),
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
            run=RunConfig(output_dir=str(root / "outputs"), response_parallel_workers=4, response_rate_limit_rpm=50),
        )

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

    def test_generation_finishes_before_judging_starts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            examples = [
                NormalizedExample(
                    id="ex1",
                    dataset_name="d",
                    provenance="canonical:test",
                    judge_mode="reference",
                    instructions="Answer",
                ),
                NormalizedExample(
                    id="ex2",
                    dataset_name="d",
                    provenance="canonical:test",
                    judge_mode="reference",
                    instructions="Answer",
                ),
            ]

            call_stages: list[str] = []

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                call_stages.append(stage)
                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": "candidate answer" if stage == "response" else '{"score": 1.0, "pass": true}',
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-request-id-{len(call_stages)}",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache-key-{len(call_stages)}"

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=([*examples], [{"dataset": "d", "path": config.data.datasets[0].path, "selected_examples": 2}]),
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
            ):
                run_module.run(config=config, progress_mode="off")

            response_indices = [i for i, stage in enumerate(call_stages) if stage == "response"]
            judge_indices = [i for i, stage in enumerate(call_stages) if stage == "judge"]
            self.assertTrue(response_indices)
            self.assertTrue(judge_indices)
            self.assertEqual(len(response_indices), len(config.candidates) * len(examples))
            self.assertEqual(len(judge_indices), len(config.candidates) * len(examples))
            self.assertLess(max(response_indices), min(judge_indices))

    def test_rubric_judge_failure_does_not_abort_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            config.run.judge_parallel_workers = 2
            example = NormalizedExample(
                id="ex1",
                dataset_name="d",
                provenance="canonical:test",
                judge_mode="rubric",
                instructions="Answer",
                rubric=[
                    {"id": "c1", "title": "Accuracy", "weight": 1.0},
                    {"id": "c2", "title": "Completeness", "weight": 1.0},
                ],
            )

            judge_calls = 0

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                nonlocal judge_calls
                if stage == "response":
                    text = "candidate answer"
                else:
                    judge_calls += 1
                    if judge_calls == 1:
                        raise RuntimeError("503 UNAVAILABLE")
                    text = '{"grade": 1, "reasoning": "ok", "criterion_id": "c2"}'

                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": text,
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-request-id-{judge_calls}",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache-key-{judge_calls}"

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=([example], [{"dataset": "d", "path": config.data.datasets[0].path, "selected_examples": 1}]),
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
                    "build_rubric_criterion_judge_messages",
                    return_value=[LLMMessage(role="user", content="judge prompt")],
                ),
                mock.patch.object(run_module, "_run_model_call", side_effect=fake_model_call),
            ):
                out_dir = run_module.run(config=config, progress_mode="off")

            judgments_path = out_dir / "judgments.jsonl"
            lines = [line for line in judgments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), len(config.candidates))
            judgments = [json.loads(line) for line in lines]
            errored = [judgment for judgment in judgments if judgment["parse_error"]]
            self.assertTrue(errored)
            calls = errored[0]["raw_judge"]["calls"]
            self.assertEqual(len(calls), 2)
            self.assertTrue(any(call.get("parse_error") for call in calls))
            self.assertTrue(any(call.get("error") for call in calls))


if __name__ == "__main__":
    unittest.main()
