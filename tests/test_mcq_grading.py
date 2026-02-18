from __future__ import annotations

import unittest

from src.judge.mcq import grade_mcq_output
from src.types import NormalizedExample


class TestMCQGrading(unittest.TestCase):
    def setUp(self) -> None:
        self.example = NormalizedExample(
            id="mcq-1",
            dataset_name="test",
            provenance="canonical:mcq",
            judge_mode="mcq",
            instructions="Q",
            metadata={"correct_choice_ids": ["C"]},
        )

    def test_exact_match_from_json_answer(self) -> None:
        out = grade_mcq_output(
            example=self.example,
            candidate_text='{"answer":"C","reasoning":"best fit"}',
            pass_threshold=0.7,
        )
        self.assertEqual(out.score, 1.0)
        self.assertTrue(out.passed)
        self.assertFalse(out.parse_error)
        self.assertEqual(out.criteria.get("exact_match"), 1.0)

    def test_incorrect_answer_scores_zero(self) -> None:
        out = grade_mcq_output(
            example=self.example,
            candidate_text='{"answer":"A","reasoning":"guess"}',
            pass_threshold=0.7,
        )
        self.assertEqual(out.score, 0.0)
        self.assertFalse(out.passed)
        self.assertFalse(out.parse_error)

    def test_non_json_output_is_parse_error(self) -> None:
        out = grade_mcq_output(
            example=self.example,
            candidate_text="I pick C",
            pass_threshold=0.7,
        )
        self.assertEqual(out.score, 0.0)
        self.assertFalse(out.passed)
        self.assertTrue(out.parse_error)


if __name__ == "__main__":
    unittest.main()
