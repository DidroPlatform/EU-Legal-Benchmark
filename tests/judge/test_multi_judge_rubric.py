from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.config import BenchmarkConfig
from src.judge.judge import (
    APEX_V1_GRADING_PROMPT_TEMPLATE,
    PRBENCH_GRADER_TEMPLATE,
    build_rubric_criterion_judge_messages,
    parse_judge_output,
    resolve_rubric_criterion_score,
)
from src.setup_checks import required_provider_names
from src.types import NormalizedExample


class TestMultiJudgeRubric(unittest.TestCase):
    def _write_dataset(self, root: Path) -> Path:
        dataset = root / "dataset.jsonl"
        dataset.write_text(
            '{"schema_version":"legal_eval_v1","id":"ex1","dataset":"d","task_type":"reference_qa","prompt":"P","reference_answers":["R"]}\n',
            encoding="utf-8",
        )
        return dataset

    def test_config_rejects_legacy_single_judge_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = self._write_dataset(root)
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
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                BenchmarkConfig.from_yaml(str(cfg_path))

    def test_config_judges_list_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = self._write_dataset(root)
            cfg_path = root / "config.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    f"""
                    providers:
                      openai:
                        api_key_env: OPENAI_API_KEY
                      vertex:
                        project: p
                        location: us-central1

                    candidates:
                      - name: c1
                        provider: openai
                        model: openai/gpt-4o-mini

                    judges:
                      - name: j1
                        provider: openai
                        model: openai/gpt-4o-mini
                      - name: j2
                        provider: vertex
                        model: vertex_ai/gemini-2.5-flash

                    data:
                      datasets:
                        - name: d
                          path: {dataset}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            cfg = BenchmarkConfig.from_yaml(str(cfg_path))
            self.assertEqual([j.name for j in cfg.judges], ["j1", "j2"])
            self.assertEqual(cfg.primary_judge.name, "j1")
            self.assertEqual(required_provider_names(cfg), {"openai", "vertex"})

    def test_build_rubric_criterion_prompt_is_single_criterion(self) -> None:
        ex = NormalizedExample(
            id="ex",
            dataset_name="d",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="Assess answer.",
            rubric=[
                {"id": "c1", "title": "Accuracy", "weight": 2.0},
                {"id": "c2", "title": "Completeness", "weight": 1.0},
            ],
        )
        messages = build_rubric_criterion_judge_messages(
            example=ex,
            model_output="candidate answer",
            criterion=ex.rubric[0],
            criterion_index=1,
            pass_threshold=0.7,
        )
        user_text = messages[1].content
        self.assertIn("Evaluate only this single rubric criterion", user_text)
        self.assertIn("c1: Accuracy", user_text)
        self.assertNotIn("c2: Completeness", user_text)
        self.assertIn('"criteria_met": bool', user_text)
        self.assertIn('"explanation": str', user_text)

    def test_apex_rubric_criterion_prompt_uses_apex_template(self) -> None:
        ex = NormalizedExample(
            id="ex-apex",
            dataset_name="apexv1",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="unused",
            messages=[],
            metadata={"policy_id": "apexv1_extended_v1"},
            rubric=[{"id": "c1", "title": "Accuracy", "description": "Answer is legally accurate."}],
        )
        messages = build_rubric_criterion_judge_messages(
            example=ex,
            model_output="candidate answer",
            criterion=ex.rubric[0],
            criterion_index=1,
            pass_threshold=0.7,
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")
        user_text = messages[0].content
        self.assertIn("Criterion to evaluate: Answer is legally accurate.", user_text)
        self.assertIn("Response to evaluate: candidate answer", user_text)
        self.assertIn('"result": <1 or 0>', user_text)
        self.assertIn('"reason": "<concise explanation>"', user_text)
        self.assertIn(
            "determine if the response satisfies this criterion",
            APEX_V1_GRADING_PROMPT_TEMPLATE,
        )

    def test_resolve_rubric_criterion_score_uses_aliases(self) -> None:
        criterion = {"id": "legal_accuracy", "title": "Legal Accuracy"}

        by_id, matched_id = resolve_rubric_criterion_score(
            criteria={"legal_accuracy": 1.0},
            criterion=criterion,
            criterion_index=1,
            fallback_score=0.0,
        )
        self.assertTrue(matched_id)
        self.assertEqual(by_id, 1.0)

        by_title, matched_title = resolve_rubric_criterion_score(
            criteria={"Legal Accuracy": 0.5},
            criterion=criterion,
            criterion_index=1,
            fallback_score=0.0,
        )
        self.assertTrue(matched_title)
        self.assertEqual(by_title, 0.5)

        fallback, matched_fallback = resolve_rubric_criterion_score(
            criteria={"other": 0.2},
            criterion=criterion,
            criterion_index=1,
            fallback_score=0.75,
        )
        self.assertFalse(matched_fallback)
        self.assertEqual(fallback, 0.75)

    def test_parse_judge_output_accepts_binary_grade_schema(self) -> None:
        out = parse_judge_output(
            raw_text='{"grade": 1, "reasoning": "criterion met", "criterion_id": "c1"}',
            fallback_pass_threshold=0.7,
        )
        self.assertEqual(out.score, 1.0)
        self.assertTrue(out.passed)
        self.assertEqual(out.rationale, "criterion met")
        self.assertEqual(out.criteria.get("overall"), 1.0)
        self.assertFalse(out.parse_error)

    def test_parse_judge_output_accepts_apex_result_schema(self) -> None:
        out = parse_judge_output(
            raw_text='{"result": 1, "reason": "criterion met"}',
            fallback_pass_threshold=0.7,
        )
        self.assertEqual(out.score, 1.0)
        self.assertTrue(out.passed)
        self.assertEqual(out.rationale, "criterion met")
        self.assertEqual(out.criteria.get("overall"), 1.0)
        self.assertFalse(out.parse_error)

    def test_parse_judge_output_accepts_apex_result_zero(self) -> None:
        out = parse_judge_output(
            raw_text='{"result": 0, "reason": "criterion not met"}',
            fallback_pass_threshold=0.7,
        )
        self.assertEqual(out.score, 0.0)
        self.assertFalse(out.passed)
        self.assertEqual(out.rationale, "criterion not met")
        self.assertEqual(out.criteria.get("overall"), 0.0)
        self.assertFalse(out.parse_error)

    def test_parse_judge_output_marks_missing_score_field_as_parse_error(self) -> None:
        out = parse_judge_output(
            raw_text='{"reason": "missing binary value"}',
            fallback_pass_threshold=0.7,
        )
        self.assertEqual(out.score, 0.0)
        self.assertFalse(out.passed)
        self.assertTrue(out.parse_error)

    def test_parse_judge_output_accepts_criteria_met_schema(self) -> None:
        out = parse_judge_output(
            raw_text='{"criteria_met": true, "explanation": "criterion met"}',
            fallback_pass_threshold=0.7,
        )
        self.assertEqual(out.score, 1.0)
        self.assertTrue(out.passed)
        self.assertEqual(out.rationale, "criterion met")
        self.assertEqual(out.criteria.get("overall"), 1.0)
        self.assertFalse(out.parse_error)

    def test_prbench_rubric_criterion_prompt_uses_template_without_candidate_duplication(self) -> None:
        ex = NormalizedExample(
            id="ex-prbench",
            dataset_name="prbench",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="unused",
            messages=[],
            metadata={"policy_id": "prbench_v1"},
            rubric=[{"id": "c1", "title": "Accuracy"}],
        )
        messages = build_rubric_criterion_judge_messages(
            example=ex,
            model_output="answer",
            criterion=ex.rubric[0],
            criterion_index=1,
            pass_threshold=0.7,
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")
        self.assertIn("# Conversation", messages[0].content)
        self.assertIn("# Rubric item", messages[0].content)
        self.assertNotIn("Candidate answer:", messages[0].content)
        self.assertIn("Accuracy", messages[0].content)
        self.assertIn("criteria_met", PRBENCH_GRADER_TEMPLATE)
