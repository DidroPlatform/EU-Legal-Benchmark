from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import run as run_module
from src.config import BenchmarkConfig, CacheConfig, DataConfig, DatasetConfig, ModelConfig, ProviderConfig, RunConfig
from src.types import JudgeResult, LLMMessage, NormalizedExample


class TestRunProgress(unittest.TestCase):
    def _build_config(self, root: Path) -> BenchmarkConfig:
        dataset_path = root / "dataset.jsonl"
        dataset_path.write_text(
            '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"Q","reference_answers":["A"]}\n',
            encoding="utf-8",
        )
        model = ModelConfig(name="cand", provider="openai", model="openai/gpt-4o-mini")
        judge = ModelConfig(name="judge", provider="openai", model="openai/gpt-4o-mini")
        return BenchmarkConfig(
            providers={"openai": ProviderConfig(api_key_env=None)},
            candidates=[model],
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
            run=RunConfig(output_dir=str(root / "outputs")),
        )

    def _example(self) -> NormalizedExample:
        return NormalizedExample(
            id="ex1",
            dataset_name="d",
            provenance="canonical:test",
            judge_mode="reference",
            instructions="Answer.",
        )

    @staticmethod
    def _fake_model_call(provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None):
        payload = {
            "provider": request.provider,
            "model": request.model,
            "text": "candidate answer" if stage == "response" else '{"score": 1.0, "pass": true}',
            "usage": {},
            "latency_s": 0.01 if stage == "response" else 0.02,
            "request_id": f"{stage}-request-id",
            "raw_response": None,
        }
        return payload, False, f"{stage}-cache-key"

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

    def test_parse_args_progress_default_and_off(self) -> None:
        args_default = run_module.parse_args([])
        self.assertEqual(args_default.progress, "log")

        args_off = run_module.parse_args(["--progress", "off"])
        self.assertEqual(args_off.progress, "off")

    def test_parse_args_progress_invalid_value(self) -> None:
        with self.assertRaises(SystemExit):
            run_module.parse_args(["--progress", "invalid"])

    def test_progress_helpers(self) -> None:
        self.assertTrue(run_module._progress_enabled("log"))
        self.assertFalse(run_module._progress_enabled("off"))
        line = run_module._progress_line(
            item="1/3",
            stage="response",
            candidate="cand",
            dataset="d",
            example="ex1",
        )
        self.assertIn("[progress]", line)
        self.assertIn("item=1/3", line)
        self.assertIn("candidate=cand", line)
        self.assertIn("dataset=d", line)
        self.assertIn("example=ex1", line)

    def test_run_emits_progress_logs_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            example = self._example()
            output = StringIO()

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=(
                        [example],
                        [{"dataset": "d", "path": config.data.datasets[0].path, "selected_examples": 1}],
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
                mock.patch.object(run_module, "_run_model_call", side_effect=self._fake_model_call),
                mock.patch.object(run_module, "parse_judge_output", side_effect=self._fake_parse_judge_output),
                mock.patch.object(run_module, "apply_weighted_rubric_score", side_effect=lambda parsed, **_: parsed),
                redirect_stdout(output),
            ):
                run_module.run(config=config, progress_mode="log")

            text = output.getvalue()
            self.assertIn("[progress] stage=start", text)
            self.assertIn("stage=response_phase_start", text)
            self.assertIn("stage=response_queued", text)
            self.assertIn("stage=response_started", text)
            self.assertIn("stage=response_done", text)
            self.assertIn("stage=response_phase_done", text)
            self.assertIn("stage=judge_phase_start", text)
            self.assertIn("stage=judge_done", text)
            self.assertIn("stage=judge_phase_done", text)

    def test_run_suppresses_progress_logs_when_off(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            example = self._example()
            output = StringIO()

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=(
                        [example],
                        [{"dataset": "d", "path": config.data.datasets[0].path, "selected_examples": 1}],
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
                mock.patch.object(run_module, "_run_model_call", side_effect=self._fake_model_call),
                mock.patch.object(run_module, "parse_judge_output", side_effect=self._fake_parse_judge_output),
                mock.patch.object(run_module, "apply_weighted_rubric_score", side_effect=lambda parsed, **_: parsed),
                redirect_stdout(output),
            ):
                run_module.run(config=config, progress_mode="off")

            text = output.getvalue()
            self.assertNotIn("[progress]", text)

    def test_generation_queue_is_not_fully_emitted_before_first_start(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            config.run.response_parallel_workers = 1
            examples = [
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
            output = StringIO()

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=(
                        examples,
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
                mock.patch.object(run_module, "_run_model_call", side_effect=self._fake_model_call),
                mock.patch.object(run_module, "parse_judge_output", side_effect=self._fake_parse_judge_output),
                mock.patch.object(run_module, "apply_weighted_rubric_score", side_effect=lambda parsed, **_: parsed),
                redirect_stdout(output),
            ):
                run_module.run(config=config, progress_mode="log")

            text = output.getvalue()
            first_started = text.find("stage=response_started")
            queue_second = text.find("item=2/2 stage=response_queued")
            self.assertGreaterEqual(first_started, 0)
            self.assertGreaterEqual(queue_second, 0)
            self.assertLess(first_started, queue_second)


if __name__ == "__main__":
    unittest.main()
