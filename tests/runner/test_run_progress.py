from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import run as run_module
from src.config import BenchmarkConfig

from tests.runner._helpers import build_runner_config, make_example, make_examples, patched_runner_env


class TestRunProgress(unittest.TestCase):
    def _build_config(self, root: Path) -> BenchmarkConfig:
        return build_runner_config(root, response_rate_limit_rpm=50)

    def _example(self):
        return make_example(instructions="Answer.")

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

            with patched_runner_env(config=config, examples=[example], run_model_call=self._fake_model_call), redirect_stdout(output):
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

            with patched_runner_env(config=config, examples=[example], run_model_call=self._fake_model_call), redirect_stdout(output):
                run_module.run(config=config, progress_mode="off")

            text = output.getvalue()
            self.assertNotIn("[progress]", text)

    def test_generation_queue_is_not_fully_emitted_before_first_start(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            config.run.response_parallel_workers = 1
            examples = make_examples(2, instructions="Answer.")
            output = StringIO()

            with patched_runner_env(config=config, examples=examples, run_model_call=self._fake_model_call), redirect_stdout(output):
                run_module.run(config=config, progress_mode="log")

            text = output.getvalue()
            first_started = text.find("stage=response_started")
            queue_second = text.find("item=2/2 stage=response_queued")
            self.assertGreaterEqual(first_started, 0)
            self.assertGreaterEqual(queue_second, 0)
            self.assertLess(first_started, queue_second)
