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
from src.types import LLMMessage, NormalizedExample


class TestRunInterrupt(unittest.TestCase):
    def _build_config(self, root: Path) -> BenchmarkConfig:
        dataset_path = root / "dataset.jsonl"
        dataset_path.write_text(
            '\n'.join(
                [
                    '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"Q1","reference_answers":["A1"]}',
                    '{"schema_version":"legal_eval_v1","id":"ex2","dataset":"d","task_type":"reference_qa","prompt":"Q2","reference_answers":["A2"]}',
                ]
            )
            + "\n",
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

    def test_generation_interrupt_writes_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self._build_config(root)
            output = StringIO()
            call_count = {"response": 0}

            def fake_model_call(
                provider, request, cache, retry_cfg, stage: str, include_raw: bool, before_attempt=None
            ):
                if stage != "response":
                    raise AssertionError("Judge phase should not run after generation interrupt.")
                call_count["response"] += 1
                if call_count["response"] == 2:
                    raise KeyboardInterrupt()
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
                redirect_stdout(output),
            ):
                out_dir = run_module.run(config=config, progress_mode="log")

            summary_path = out_dir / "summary.json"
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            self.assertEqual(summary.get("run_status"), "interrupted")
            self.assertEqual(summary.get("interrupted_stage"), "generation")

            responses = [line for line in (out_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines() if line]
            judgments = [line for line in (out_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(responses), 1)
            self.assertEqual(len(judgments), 0)

            text = output.getvalue()
            self.assertIn("stage=response_phase_interrupted", text)


if __name__ == "__main__":
    unittest.main()
