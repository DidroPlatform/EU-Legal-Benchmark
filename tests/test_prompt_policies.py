from __future__ import annotations

import unittest

from src.prompting.templates import build_candidate_messages
from src.types import LLMMessage, NormalizedExample


class TestPromptPolicies(unittest.TestCase):
    def test_prbench_policy_has_no_default_system_prompt(self) -> None:
        ex = NormalizedExample(
            id="pr-1",
            dataset_name="d",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="Prompt",
            messages=[],
            metadata={"policy_id": "prbench_v1"},
        )
        msgs = build_candidate_messages(ex, "DEFAULT_SYS")
        self.assertTrue(msgs)
        self.assertEqual(msgs[0].role, "user")
        self.assertFalse(any(m.role == "system" for m in msgs))

    def test_apex_policy_adds_domain_and_attachment_block(self) -> None:
        ex = NormalizedExample(
            id="ax-1",
            dataset_name="d",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="Prompt",
            messages=[],
            metadata={
                "policy_id": "apexv1_extended_v1",
                "domain": "Legal",
                "attachments": [{"path": "documents/1382/file.pdf", "kind": "pdf"}],
                "attachment_contents": [
                    {
                        "path": "documents/1382/file.pdf",
                        "kind": "pdf",
                        "text": "Convention body text",
                    }
                ],
            },
        )
        msgs = build_candidate_messages(ex, "DEFAULT_SYS")
        user_text = "\n".join(m.content for m in msgs if m.role == "user")
        self.assertIn("Domain: Legal", user_text)
        self.assertIn("==== Attached files content", user_text)
        self.assertIn("documents/1382/file.pdf (pdf)", user_text)
        self.assertIn("Convention body text", user_text)

    def test_apex_policy_merges_policy_guidance_into_first_user_message(self) -> None:
        ex = NormalizedExample(
            id="ax-2",
            dataset_name="d",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="Prompt",
            messages=[LLMMessage(role="user", content="Prompt")],
            metadata={
                "policy_id": "apexv1_extended_v1",
                "domain": "Legal",
                "attachment_contents": [
                    {"path": "documents/1382/file.pdf", "kind": "pdf", "text": "Body"}
                ],
            },
        )
        msgs = build_candidate_messages(ex, "DEFAULT_SYS")
        self.assertEqual(sum(1 for m in msgs if m.role == "user"), 1)
        self.assertIn("Prompt", msgs[0].content)
        self.assertIn("Domain: Legal", msgs[0].content)
        self.assertIn("==== Attached files content: ====", msgs[0].content)

    def test_lexam_open_policy_instructions_present(self) -> None:
        ex = NormalizedExample(
            id="lx-1",
            dataset_name="d",
            provenance="canonical:reference_qa",
            judge_mode="reference",
            instructions="Frage",
            messages=[],
            metadata={"policy_id": "lexam_oq_v1"},
        )
        msgs = build_candidate_messages(ex, "DEFAULT_SYS")
        self.assertTrue(any(m.role == "system" for m in msgs))
        user_text = "\n".join(m.content for m in msgs if m.role == "user")
        self.assertIn("same language", user_text.lower())
        self.assertIn("Abs.", user_text)
        self.assertIn("Ziff.", user_text)
        self.assertIn("lit.", user_text)
        self.assertIn("missing", user_text.lower())

    def test_lar_echr_policy_includes_default_system_and_lar_guidance(self) -> None:
        ex = NormalizedExample(
            id="lar-1",
            dataset_name="d",
            provenance="canonical:mcq",
            judge_mode="mcq",
            instructions="Select the best continuation.",
            messages=[],
            metadata={"policy_id": "lar_echr_mcq_v1"},
        )
        msgs = build_candidate_messages(ex, "DEFAULT_SYS")
        self.assertTrue(any(m.role == "system" for m in msgs))
        user_text = "\n".join(m.content for m in msgs if m.role == "user")
        self.assertIn("ECHR argument-continuation", user_text)
        self.assertIn("Return only valid JSON", user_text)


if __name__ == "__main__":
    unittest.main()
