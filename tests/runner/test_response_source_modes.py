from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run as run_module
from src.config import BenchmarkConfig, CacheConfig, DataConfig, DatasetConfig, ModelConfig, ProviderConfig, RunConfig
from src.types import JudgeResult, NormalizedExample


class TestResponseSourceModes(unittest.TestCase):
    def _build_config(self, root: Path, *, final_source: str, prefilled_path: str | None = None) -> BenchmarkConfig:
        dataset_path = root / "dataset.jsonl"
        dataset_path.write_text(
            '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"Q","reference_answers":["A"]}\n',
            encoding="utf-8",
        )
        candidate = ModelConfig(name="cand_a", provider="openai", model="openai/gpt-4o-mini")
        judge = ModelConfig(name="judge", provider="openai", model="openai/gpt-4o-mini")
        run_cfg = RunConfig(
            output_dir=str(root / "outputs"),
            response_parallel_workers=1,
            response_rate_limit_rpm=50,
            final_response_source=final_source,
            prefilled_responses_path=prefilled_path,
        )
        return BenchmarkConfig(
            providers={"openai": ProviderConfig(api_key_env=None)},
            candidates=[candidate],
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
            run=run_cfg,
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

    def test_sampled_source_executes_response_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root, final_source="sampled")
            examples = [
                NormalizedExample(
                    id="ex1",
                    dataset_name="d",
                    provenance="canonical:test",
                    judge_mode="reference",
                    instructions="Q",
                )
            ]

            call_stages: list[str] = []

            def fake_model_call(provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None):
                call_stages.append(stage)
                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": "candidate answer" if stage == "response" else '{"score": 1.0, "pass": true}',
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-req",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache"

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=(examples, [{"dataset": "d", "path": config.data.datasets[0].path, "selected_examples": 1}]),
                ),
                mock.patch.object(run_module, "required_provider_names", return_value={"openai"}),
                mock.patch.object(run_module, "build_provider", return_value=object()),
                mock.patch.object(run_module, "_run_model_call", side_effect=fake_model_call),
                mock.patch.object(run_module, "parse_judge_output", side_effect=self._fake_parse_judge_output),
                mock.patch.object(run_module, "apply_weighted_rubric_score", side_effect=lambda parsed, **_: parsed),
            ):
                run_module.run(config=config, progress_mode="off")

            self.assertIn("response", call_stages)

    def test_prefilled_source_skips_response_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prefilled_path = root / "prefilled.jsonl"
            prefilled_path.write_text(
                '{"example_id":"ex1","candidate_name":"cand_a","response_text":"prefilled answer"}\n',
                encoding="utf-8",
            )
            config = self._build_config(root, final_source="prefilled", prefilled_path=str(prefilled_path))
            examples = [
                NormalizedExample(
                    id="ex1",
                    dataset_name="d",
                    provenance="canonical:test",
                    judge_mode="reference",
                    instructions="Q",
                )
            ]

            call_stages: list[str] = []

            def fake_model_call(provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None):
                call_stages.append(stage)
                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": '{"score": 1.0, "pass": true}',
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-req",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache"

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=(examples, [{"dataset": "d", "path": config.data.datasets[0].path, "selected_examples": 1}]),
                ),
                mock.patch.object(run_module, "required_provider_names", return_value={"openai"}),
                mock.patch.object(run_module, "build_provider", return_value=object()),
                mock.patch.object(run_module, "_run_model_call", side_effect=fake_model_call),
                mock.patch.object(run_module, "parse_judge_output", side_effect=self._fake_parse_judge_output),
                mock.patch.object(run_module, "apply_weighted_rubric_score", side_effect=lambda parsed, **_: parsed),
            ):
                out_dir = run_module.run(config=config, progress_mode="off")

            self.assertNotIn("response", call_stages)
            responses = (out_dir / "responses.jsonl").read_text(encoding="utf-8")
            self.assertIn('"response_source": "prefilled"', responses)
            self.assertIn("prefilled answer", responses)

    def test_prefilled_source_missing_entry_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prefilled_path = root / "prefilled.jsonl"
            prefilled_path.write_text("", encoding="utf-8")
            config = self._build_config(root, final_source="prefilled", prefilled_path=str(prefilled_path))
            examples = [
                NormalizedExample(
                    id="ex1",
                    dataset_name="d",
                    provenance="canonical:test",
                    judge_mode="reference",
                    instructions="Q",
                )
            ]

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=(examples, [{"dataset": "d", "path": config.data.datasets[0].path, "selected_examples": 1}]),
                ),
                mock.patch.object(run_module, "required_provider_names", return_value={"openai"}),
                mock.patch.object(run_module, "build_provider", return_value=object()),
            ):
                with self.assertRaises(ValueError) as ctx:
                    run_module.run(config=config, progress_mode="off")
            self.assertIn("Missing prefilled responses", str(ctx.exception))

    def test_part_of_conversation_source_skips_response_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            previous_output = root / "responses.jsonl"
            previous_output.write_text(
                '{"example_id":"ex1","candidate_name":"cand_a","response_text":"reused answer"}\n',
                encoding="utf-8",
            )
            config = self._build_config(root, final_source="part_of_conversation")
            config.run.previous_output_path = str(previous_output)
            examples = [
                NormalizedExample(
                    id="ex1",
                    dataset_name="d",
                    provenance="canonical:test",
                    judge_mode="reference",
                    instructions="Q",
                )
            ]

            call_stages: list[str] = []

            def fake_model_call(provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None):
                call_stages.append(stage)
                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": '{"score": 1.0, "pass": true}',
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-req",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache"

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=(examples, [{"dataset": "d", "path": config.data.datasets[0].path, "selected_examples": 1}]),
                ),
                mock.patch.object(run_module, "required_provider_names", return_value={"openai"}),
                mock.patch.object(run_module, "build_provider", return_value=object()),
                mock.patch.object(run_module, "_run_model_call", side_effect=fake_model_call),
                mock.patch.object(run_module, "parse_judge_output", side_effect=self._fake_parse_judge_output),
                mock.patch.object(run_module, "apply_weighted_rubric_score", side_effect=lambda parsed, **_: parsed),
            ):
                out_dir = run_module.run(config=config, progress_mode="off")

            self.assertNotIn("response", call_stages)
            responses = (out_dir / "responses.jsonl").read_text(encoding="utf-8")
            self.assertIn('"response_source": "part_of_conversation"', responses)
            self.assertIn("reused answer", responses)

