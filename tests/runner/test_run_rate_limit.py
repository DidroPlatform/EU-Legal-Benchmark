from __future__ import annotations

import json
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

import run as run_module
from src.config import BenchmarkConfig, DataConfig, DatasetConfig, ModelConfig, ProviderConfig, RunConfig

from tests.runner._helpers import (
    build_runner_config,
    make_example,
    make_examples,
    patched_runner_env,
)


class TestRunRateLimit(unittest.TestCase):
    def test_config_rejects_response_rpm_over_50(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"P","reference_answers":["R"]}\n',
                encoding="utf-8",
            )
            cfg_path = root / "config.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    f"""
                    providers:
                      openai:
                        api_key_env: OPENAI_API_KEY

                    candidates:
                      - name: c1
                        provider: openai
                        model: openai/gpt-4o-mini

                    judges:
                      - name: j1
                        provider: openai
                        model: openai/gpt-4o-mini

                    data:
                      datasets:
                        - name: d
                          path: {dataset}

                    run:
                      response_rate_limit_rpm: 51
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                BenchmarkConfig.from_yaml(str(cfg_path))

    def test_per_minute_rate_limiter_enforces_spacing(self) -> None:
        now = 0.0
        admissions: list[float] = []

        def monotonic() -> float:
            return now

        def sleep(seconds: float) -> None:
            nonlocal now
            now += seconds

        limiter = run_module.PerMinuteRateLimiter(50, monotonic_fn=monotonic, sleep_fn=sleep)
        for _ in range(5):
            limiter.wait()
            admissions.append(now)

        expected_interval = 60.0 / 50.0
        deltas = [admissions[idx] - admissions[idx - 1] for idx in range(1, len(admissions))]
        for delta in deltas:
            self.assertGreaterEqual(delta, expected_interval - 1e-9)

    def test_config_rejects_provider_specific_rpm_for_unknown_provider(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(
                '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"Q","reference_answers":["A"]}\n',
                encoding="utf-8",
            )
            model = ModelConfig(name="cand", provider="openai", model="openai/gpt-4o-mini")
            judge = ModelConfig(name="judge", provider="openai", model="openai/gpt-4o-mini")
            with self.assertRaises(ValueError):
                BenchmarkConfig(
                    providers={"openai": ProviderConfig(api_key_env=None)},
                    candidates=[model],
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
                    run=RunConfig(provider_response_rate_limit_rpm={"nim": 20}),
                ).validate()

    def test_config_requires_previous_output_for_part_of_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(
                '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"Q","reference_answers":["A"]}\n',
                encoding="utf-8",
            )
            model = ModelConfig(name="cand", provider="openai", model="openai/gpt-4o-mini")
            judge = ModelConfig(name="judge", provider="openai", model="openai/gpt-4o-mini")
            with self.assertRaises(ValueError) as ctx:
                BenchmarkConfig(
                    providers={"openai": ProviderConfig(api_key_env=None)},
                    candidates=[model],
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
                    run=RunConfig(final_response_source="part_of_conversation"),
                ).validate()
            self.assertIn("run.previous_output_path", str(ctx.exception))

    def test_programmatic_validate_requires_non_empty_judges(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(
                '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"Q","reference_answers":["A"]}\n',
                encoding="utf-8",
            )
            candidate = ModelConfig(name="cand", provider="openai", model="openai/gpt-4o-mini")
            config = BenchmarkConfig(
                providers={"openai": ProviderConfig(api_key_env=None)},
                candidates=[candidate],
                judges=[],
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
            )

            with self.assertRaises(ValueError):
                config.validate()

    def test_run_validates_config_for_programmatic_part_of_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            config.run.final_response_source = "part_of_conversation"
            config.run.previous_output_path = None

            with self.assertRaises(ValueError) as ctx:
                run_module.run(config=config, progress_mode="off")
            self.assertIn("run.previous_output_path", str(ctx.exception))

    def test_config_rejects_google_responses_api_for_sampled_generation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(
                '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"Q","reference_answers":["A"]}\n',
                encoding="utf-8",
            )
            model = ModelConfig(name="cand", provider="google_genai", model="gemini-flash-lite-latest")
            judge = ModelConfig(name="judge", provider="google_genai", model="gemini-flash-lite-latest")
            with self.assertRaises(ValueError) as ctx:
                BenchmarkConfig(
                    providers={"google_genai": ProviderConfig(api_key_env=None)},
                    candidates=[model],
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
                    run=RunConfig(response_api="responses"),
                ).validate()
            self.assertIn("does not support run.response_api='responses'", str(ctx.exception))

    def _build_config(self, root: Path) -> BenchmarkConfig:
        return build_runner_config(root, candidate_count=1)

    def test_limit_override_keeps_dataset_selected_examples_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            examples = make_examples(2)

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": "candidate answer" if stage == "response" else '{"score": 1.0, "pass": true}',
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-request-id",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache-key"

            with patched_runner_env(config=config, examples=examples, run_model_call=fake_model_call):
                out_dir = run_module.run(config=config, limit_override=1, progress_mode="off")

            with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)

            self.assertEqual(summary.get("selected_examples"), 1)
            datasets = summary.get("datasets", [])
            self.assertEqual(len(datasets), 1)
            self.assertEqual(datasets[0].get("selected_examples"), 1)

    def test_generation_uses_parallel_workers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            examples = make_examples(4)

            lock = threading.Lock()
            inflight = 0
            max_inflight = 0

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                nonlocal inflight, max_inflight
                if stage == "response":
                    with lock:
                        inflight += 1
                        max_inflight = max(max_inflight, inflight)
                    time.sleep(0.05)
                    with lock:
                        inflight -= 1
                    text = "candidate answer"
                else:
                    text = '{"score": 1.0, "pass": true}'

                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": text,
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-request-id",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache-key"

            with patched_runner_env(
                config=config,
                examples=examples,
                run_model_call=fake_model_call,
                limiter_wait_return=None,
            ):
                run_module.run(config=config, progress_mode="off")

            self.assertGreater(max_inflight, 1)

    def test_generation_before_attempt_waits_on_global_and_provider_limiters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            config.run.provider_response_rate_limit_rpm = {"openai": 20}
            example = make_example()

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                if before_attempt is not None:
                    before_attempt(1)
                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": "candidate answer" if stage == "response" else '{"score": 1.0, "pass": true}',
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-request-id",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache-key"

            with patched_runner_env(
                config=config,
                examples=[example],
                run_model_call=fake_model_call,
                limiter_wait_return=None,
            ) as handles:
                run_module.run(config=config, progress_mode="off")

            self.assertEqual(handles["wait"].call_count, 2)

    def test_judging_uses_parallel_workers_across_items(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            config.run.judge_parallel_workers = 4
            examples = make_examples(4)

            lock = threading.Lock()
            inflight_judge = 0
            max_inflight_judge = 0

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                nonlocal inflight_judge, max_inflight_judge
                if stage == "judge":
                    with lock:
                        inflight_judge += 1
                        max_inflight_judge = max(max_inflight_judge, inflight_judge)
                    time.sleep(0.05)
                    with lock:
                        inflight_judge -= 1
                    text = '{"score": 1.0, "pass": true}'
                else:
                    text = "candidate answer"

                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": text,
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": f"{stage}-request-id",
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache-key"

            with patched_runner_env(
                config=config,
                examples=examples,
                run_model_call=fake_model_call,
                limiter_wait_return=None,
            ):
                run_module.run(config=config, progress_mode="off")

            self.assertGreater(max_inflight_judge, 1)

    def test_judgments_are_stably_ordered_by_display_index_when_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            config.candidates = [
                ModelConfig(name="cand_a", provider="openai", model="openai/gpt-4o-mini"),
                ModelConfig(name="cand_b", provider="openai", model="openai/gpt-4o-mini"),
            ]
            config.run.judge_parallel_workers = 4
            examples = make_examples(2)
            judge_call_count = 0
            lock = threading.Lock()

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                if stage == "judge":
                    nonlocal judge_call_count
                    with lock:
                        judge_call_count += 1
                        judge_call_number = judge_call_count
                    if judge_call_number == 1:
                        time.sleep(0.08)
                    elif judge_call_number == 2:
                        time.sleep(0.04)
                    text = '{"score": 1.0, "pass": true}'
                else:
                    text = "candidate answer"

                payload = {
                    "provider": request.provider,
                    "model": request.model,
                    "text": text,
                    "usage": {},
                    "latency_s": 0.01,
                    "request_id": request.request_id,
                    "raw_response": None,
                }
                return payload, False, f"{stage}-cache-key"

            with patched_runner_env(
                config=config,
                examples=examples,
                run_model_call=fake_model_call,
                limiter_wait_return=None,
            ):
                out_dir = run_module.run(config=config, progress_mode="off")

            judgments = [
                json.loads(line)
                for line in (out_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ordered_pairs = [(row["candidate_name"], row["example_id"]) for row in judgments]
            self.assertEqual(
                ordered_pairs,
                [
                    ("cand_a", "ex1"),
                    ("cand_a", "ex2"),
                    ("cand_b", "ex1"),
                    ("cand_b", "ex2"),
                ],
            )
