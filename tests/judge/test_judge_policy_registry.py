from __future__ import annotations

import unittest

from src.judge.policies.registry import get_judge_policy_handler


class TestJudgePolicyRegistry(unittest.TestCase):
    def test_registry_routes_known_policy_ids(self) -> None:
        self.assertEqual(get_judge_policy_handler("prbench_v1").policy_id, "prbench_v1")
        self.assertEqual(get_judge_policy_handler("apexv1_extended_v1").policy_id, "apexv1_extended_v1")
        self.assertEqual(get_judge_policy_handler("lexam_oq_v1").policy_id, "lexam_oq_v1")

    def test_registry_falls_back_to_default_handler(self) -> None:
        self.assertEqual(get_judge_policy_handler(None).policy_id, "default_v1")
        self.assertEqual(get_judge_policy_handler("unknown_policy").policy_id, "default_v1")

