from __future__ import annotations

import unittest

from src.config import DatasetConfig
from src.data.loader import normalize_row
from src.judge.judge import (
    LEXAM_JUDGE_SYSTEM,
    LEXAM_JUDGE_USER_PROMPT,
    apply_policy_score_postprocessing,
    build_judge_messages,
    parse_judge_output,
)
from src.types import NormalizedExample


class TestLEXamParity(unittest.TestCase):
    def test_lexam_judge_constants_exact_match(self) -> None:
        expected_system = "Act as a Judge specializing in the evaluation of Swiss law schools exams. Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning."
        expected_user = """Goal:
Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning.

Context:
You will be provided with a response (labeled: Model's Answer) to a law school exam question (labeled: Question) and a reference answer (labeled: Reference Answer). 

Return format:
    After reviewing the response:
    1. Explanation: Briefly explain your reasoning regarding how the response conforms to or deviates from the reference answer. 
    2. Constructive feedback: Additionally, provide neutral, constructive feedback and corrections in the style of a university professor.
    3. Correctness score: Assign a final correctness score on a scale from 0.0 to 1.0 (in increments of 0.1). This score should reflect the extent to which the response satisfies the reference answer, where 
        - 1.0 = complete fulfillment (100%) 
        - lower scores reflect proportionate shortfalls (e.g. 0.5 = 50% fulfillment). 
        - The correctness score will be provided in the JSON output format specified below.

Warnings:
    - In some cases, the reference answer may include only keywords or factual elements to be examined, along with (+), (-) or (+/-). Respect these indications when determining correctness:
        - (+) means the element must be affirmed.
        - (–) means the element must be denied.
        - (-/+) indicates that arguments in either direction are acceptable if legally sound.
    - Deviations or additional elements not found in the reference answer should generally be penalized unless you are certain they are legally correct and relevant. Assume the reference answer includes all information necessary for a perfect response.
    - The reference answer may contain citations (e.g., from books or law review articles), which the response does not need to replicate. However, statutes should be cited precisely, specifying Abs., Ziff., or lit. whenever applicable.
    - If the reference answer includes separate sub-points, use these for proportional scoring guidance (e.g., addressing 2 out of 4 sub-points correctly equals approximately a 0.5 score).
Judge the below case, give the brief reasoning process and the final grade.
"""
        self.assertEqual(LEXAM_JUDGE_SYSTEM, expected_system)
        self.assertEqual(LEXAM_JUDGE_USER_PROMPT, expected_user)

    def test_build_judge_messages_uses_lexam_branch(self) -> None:
        ex = NormalizedExample(
            id="lx-judge-1",
            dataset_name="d",
            provenance="canonical:reference_qa",
            judge_mode="reference",
            instructions="Frage?",
            reference_answer="Referenz",
            metadata={"policy_id": "lexam_oq_v1"},
        )
        msgs = build_judge_messages(ex, "Antwort", pass_threshold=0.7)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[0].content, LEXAM_JUDGE_SYSTEM)
        self.assertEqual(msgs[1].role, "user")
        user_text = msgs[1].content
        self.assertIn("Goal:", user_text)
        self.assertIn("Context:", user_text)
        self.assertIn("Warnings:", user_text)
        self.assertIn('{"score": <float 0.0-1.0 step 0.1>', user_text)
        self.assertIn("```Frage?```", user_text)
        self.assertIn("```Referenz```", user_text)
        self.assertIn("```[Antwort]```", user_text)

    def test_lexam_loader_mcq_has_no_choices_header_or_generic_suffix(self) -> None:
        row = {
            "schema_version": "legal_eval_v1",
            "id": "lx-mcq-1",
            "dataset": "lexam",
            "task_type": "mcq",
            "prompt": "Welche Norm gilt?",
            "choices": [
                {"id": "A", "text": "Erste"},
                {"id": "B", "text": "Zweite"},
            ],
            "correct_choice_ids": ["B"],
            "metadata": {"policy_id": "lexam_mcq_v1"},
        }
        dataset = DatasetConfig(name="d", path="unused.jsonl")
        out = normalize_row(row, dataset)
        self.assertIn("A. Erste", out.instructions)
        self.assertIn("B. Zweite", out.instructions)
        self.assertNotIn("Choices:", out.instructions)
        self.assertNotIn("Answer with the best option and brief reasoning.", out.instructions)

    def test_lexam_score_quantization_round_half_up_and_clamp(self) -> None:
        ex = NormalizedExample(
            id="lx-q-1",
            dataset_name="d",
            provenance="canonical:reference_qa",
            judge_mode="reference",
            instructions="Q",
            metadata={"policy_id": "lexam_oq_v1"},
        )

        parsed_a = parse_judge_output(
            raw_text='{"score": 0.84, "pass": true, "rationale": "x", "criteria": {"overall": 0.84}}',
            fallback_pass_threshold=0.7,
        )
        scored_a = apply_policy_score_postprocessing(parsed_a, ex, pass_threshold=0.7)
        self.assertEqual(scored_a.score, 0.8)
        self.assertEqual(scored_a.criteria["overall"], 0.8)
        self.assertTrue(scored_a.passed)

        parsed_b = parse_judge_output(
            raw_text='{"score": 0.85, "pass": false, "rationale": "x", "criteria": {"overall": 0.85}}',
            fallback_pass_threshold=0.7,
        )
        scored_b = apply_policy_score_postprocessing(parsed_b, ex, pass_threshold=0.7)
        self.assertEqual(scored_b.score, 0.9)
        self.assertEqual(scored_b.criteria["overall"], 0.9)
        self.assertTrue(scored_b.passed)

        parsed_c = parse_judge_output(
            raw_text='{"score": 1.04, "pass": false, "rationale": "x", "criteria": {"overall": 1.0}}',
            fallback_pass_threshold=0.7,
        )
        scored_c = apply_policy_score_postprocessing(parsed_c, ex, pass_threshold=0.7)
        self.assertEqual(scored_c.score, 1.0)
        self.assertEqual(scored_c.criteria["overall"], 1.0)
        self.assertTrue(scored_c.passed)

