from __future__ import annotations

import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import run as run_module
from src.config import BenchmarkConfig, CacheConfig, DataConfig, DatasetConfig, ModelConfig, ProviderConfig, RunConfig
from src.types import JudgeResult, LLMMessage, NormalizedExample


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

                    judge:
                      name: j1
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
                    run=RunConfig(provider_response_rate_limit_rpm={"nim": 20}),
                ).validate()

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
            run=RunConfig(
                output_dir=str(root / "outputs"),
                response_parallel_workers=4,
                response_rate_limit_rpm=50,
            ),
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

    def test_generation_uses_parallel_workers(self) -> None:
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
                NormalizedExample(
                    id="ex3",
                    dataset_name="d",
                    provenance="canonical:test",
                    judge_mode="reference",
                    instructions="Answer",
                ),
                NormalizedExample(
                    id="ex4",
                    dataset_name="d",
                    provenance="canonical:test",
                    judge_mode="reference",
                    instructions="Answer",
                ),
            ]

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

            with (
                mock.patch.object(run_module, "_validate_canonical_inputs", return_value=None),
                mock.patch.object(
                    run_module,
                    "_load_all_examples",
                    return_value=([*examples], [{"dataset": "d", "path": config.data.datasets[0].path, "selected_examples": 4}]),
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
                mock.patch.object(run_module.PerMinuteRateLimiter, "wait", return_value=None),
            ):
                run_module.run(config=config, progress_mode="off")

            self.assertGreater(max_inflight, 1)

    def test_generation_before_attempt_waits_on_global_and_provider_limiters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            config.run.provider_response_rate_limit_rpm = {"openai": 20}
            example = NormalizedExample(
                id="ex1",
                dataset_name="d",
                provenance="canonical:test",
                judge_mode="reference",
                instructions="Answer",
            )

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
                    "build_judge_messages",
                    return_value=[LLMMessage(role="user", content="judge prompt")],
                ),
                mock.patch.object(run_module, "_run_model_call", side_effect=fake_model_call),
                mock.patch.object(run_module, "parse_judge_output", side_effect=self._fake_parse_judge_output),
                mock.patch.object(run_module, "apply_weighted_rubric_score", side_effect=lambda parsed, **_: parsed),
                mock.patch.object(run_module.PerMinuteRateLimiter, "wait", return_value=None) as wait_mock,
            ):
                run_module.run(config=config, progress_mode="off")

            self.assertEqual(wait_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
