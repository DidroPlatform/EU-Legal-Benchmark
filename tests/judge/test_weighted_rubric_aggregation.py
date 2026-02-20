from __future__ import annotations

import unittest

from src.judge.judge import apply_weighted_rubric_score
from src.types import JudgeResult, NormalizedExample


class TestWeightedRubricAggregation(unittest.TestCase):
    def test_positive_weights_are_weighted_average(self) -> None:
        ex = NormalizedExample(
            id="r1",
            dataset_name="d",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="Q",
            rubric=[{"id": "c1", "title": "A", "weight": 2.0}, {"id": "c2", "title": "B", "weight": 1.0}],
        )
        parsed = JudgeResult(
            score=0.0,
            passed=False,
            rationale="",
            criteria={"c1": 1.0, "c2": 0.0},
            raw={},
        )
        out = apply_weighted_rubric_score(parsed, ex, 0.7)
        self.assertAlmostEqual(out.score, 2.0 / 3.0, places=6)
        self.assertFalse(out.passed)
        self.assertIn("deterministic_rubric_aggregation", out.raw)
        agg = out.raw["deterministic_rubric_aggregation"]
        self.assertAlmostEqual(agg["raw_sum"], 2.0, places=6)
        self.assertAlmostEqual(agg["normalized_points"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(agg["clipped_points"], 2.0 / 3.0, places=6)

    def test_mixed_positive_negative_weights_normalize_to_unit_interval(self) -> None:
        ex = NormalizedExample(
            id="r2",
            dataset_name="d",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="Q",
            rubric=[{"id": "good", "title": "Good", "weight": 8.0}, {"id": "bad", "title": "Bad", "weight": -5.0}],
        )
        # both met: raw_sum=3, min_raw=-5, max_raw=8 => normalized 8/13
        parsed = JudgeResult(
            score=0.0,
            passed=False,
            rationale="",
            criteria={"good": 1.0, "bad": 1.0},
            raw={},
        )
        out = apply_weighted_rubric_score(parsed, ex, 0.7)
        self.assertAlmostEqual(out.score, 8.0 / 13.0, places=6)
        self.assertFalse(out.passed)

        # good met, bad not met: raw_sum=8 => normalized 1.0
        parsed2 = JudgeResult(
            score=0.0,
            passed=False,
            rationale="",
            criteria={"good": 1.0, "bad": 0.0},
            raw={},
        )
        out2 = apply_weighted_rubric_score(parsed2, ex, 0.7)
        self.assertAlmostEqual(out2.score, 1.0, places=6)
        self.assertTrue(out2.passed)

