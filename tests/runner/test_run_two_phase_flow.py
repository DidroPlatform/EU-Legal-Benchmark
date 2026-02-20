from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import run as run_module
from src.config import BenchmarkConfig
from src.types import JudgeResult, LLMMessage

from tests.runner._helpers import (
    build_runner_config,
    make_example,
    make_examples,
    patched_runner_env,
)


class TestRunTwoPhaseFlow(unittest.TestCase):
    def _build_config(self, root: Path) -> BenchmarkConfig:
        return build_runner_config(root, candidate_names=["cand_a", "cand_b"])

    def test_generation_finishes_before_judging_starts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            examples = make_examples(2)

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

            with patched_runner_env(config=config, examples=examples, run_model_call=fake_model_call):
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
            example = make_example(
                judge_mode="rubric",
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

            with patched_runner_env(
                config=config,
                examples=[example],
                run_model_call=fake_model_call,
                parse_judge_output=None,
                apply_weighted_rubric_score=None,
                rubric_judge_messages=[LLMMessage(role="user", content="judge prompt")],
            ):
                out_dir = run_module.run(config=config, progress_mode="off")

            with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertGreaterEqual(summary.get("num_failures", 0), 1)
            self.assertTrue(any(item.get("criterion_id") for item in summary.get("failed_items", [])))

            judgments_path = out_dir / "judgments.jsonl"
            lines = [line for line in judgments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), len(config.candidates))
            judgments = [json.loads(line) for line in lines]
            errored = [judgment for judgment in judgments if judgment["parse_error"]]
            self.assertTrue(errored)
            self.assertTrue(all(j["score"] == 0.0 for j in errored))
            self.assertTrue(all(j["pass"] is False for j in errored))
            calls = errored[0]["raw_judge"]["calls"]
            self.assertEqual(len(calls), 2)
            self.assertTrue(any(call.get("parse_error") for call in calls))
            self.assertTrue(any(call.get("error") for call in calls))

    def test_single_judge_provider_failure_is_recorded_in_failed_items(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            example = make_example()

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                if stage == "judge":
                    raise RuntimeError("judge provider unavailable")
                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": "candidate answer",
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": "response-request-id",
                    "raw_response": None,
                }
                return payload, False, "response-cache-key"

            with patched_runner_env(config=config, examples=[example], run_model_call=fake_model_call):
                out_dir = run_module.run(config=config, progress_mode="off")

            with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("num_failures"), 2)
            errors = [item.get("error", "") for item in summary.get("failed_items", [])]
            self.assertTrue(any("judge provider unavailable" in err for err in errors))
            self.assertTrue(all(item.get("stage") in {"judge", "response"} for item in summary.get("failed_items", [])))

    def test_parse_only_judge_error_not_added_to_failed_items(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            example = make_example()

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": "candidate answer" if stage == "response" else '{"score": "not-a-number"}',
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-request-id",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache-key"

            def parse_with_parse_error(raw_text: str, fallback_pass_threshold: float) -> JudgeResult:
                del raw_text, fallback_pass_threshold
                return JudgeResult(
                    score=0.0,
                    passed=False,
                    rationale="parse failed",
                    criteria={},
                    raw={"reason": "parse"},
                    parse_error=True,
                )

            with patched_runner_env(
                config=config,
                examples=[example],
                run_model_call=fake_model_call,
                parse_judge_output=parse_with_parse_error,
            ):
                out_dir = run_module.run(config=config, progress_mode="off")

            with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("num_failures"), 0)
            self.assertEqual(summary.get("failed_items"), [])

    def test_parallel_judging_preserves_progress_and_row_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            config.run.judge_parallel_workers = 4
            examples = make_examples(2)
            output = StringIO()
            judge_call_count = 0

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                nonlocal judge_call_count
                if stage == "judge":
                    judge_call_count += 1
                    if judge_call_count == 1:
                        # Force first submitted judge call to complete later.
                        import time

                        time.sleep(0.05)
                    text = '{"score": 1.0, "pass": true}'
                else:
                    text = "candidate answer"
                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": text,
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-request-id-{judge_call_count}",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache-key-{judge_call_count}"

            with patched_runner_env(config=config, examples=examples, run_model_call=fake_model_call), redirect_stdout(output):
                out_dir = run_module.run(config=config, progress_mode="log")

            text = output.getvalue()
            expected_items = len(config.candidates) * len(examples)
            self.assertEqual(text.count("stage=judge_done"), expected_items)

            judgments = [
                line
                for line in (out_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(judgments), expected_items)

    def test_unexpected_judge_handler_exception_is_fail_closed_and_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            example = make_example()

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": "candidate answer",
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": "response-request-id",
                    "raw_response": None,
                }
                return payload, False, "response-cache-key"

            with (
                patched_runner_env(
                    config=config,
                    examples=[example],
                    run_model_call=fake_model_call,
                    parse_judge_output=None,
                    apply_weighted_rubric_score=None,
                ),
                mock.patch("src.runner.judging._handle_single_judge", side_effect=RuntimeError("unexpected-judge-boom")),
            ):
                out_dir = run_module.run(config=config, progress_mode="off")

            judgments = [
                json.loads(line)
                for line in (out_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(judgments), len(config.candidates))
            self.assertTrue(all(j["parse_error"] for j in judgments))
            self.assertTrue(all(j["score"] == 0.0 for j in judgments))

            with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("run_status"), "completed")
            self.assertEqual(summary.get("num_failures"), len(config.candidates))
            self.assertTrue(any("unexpected-judge-boom" in item.get("error", "") for item in summary.get("failed_items", [])))
