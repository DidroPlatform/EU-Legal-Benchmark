from __future__ import annotations

import unittest

from src.prompting.templates import (
    APEX_V1_GENERATION_SYSTEM_PROMPT,
    LEXAM_MCQ_PROMPT_LETTERS,
    LEXAM_QA_PROMPT,
    build_candidate_messages,
)
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

    def test_prbench_messages_are_forwarded_without_extra_system_prompt(self) -> None:
        ex = NormalizedExample(
            id="pr-1b",
            dataset_name="d",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="unused",
            messages=[
                LLMMessage(role="user", content="Q1"),
                LLMMessage(role="assistant", content="A1"),
                LLMMessage(role="user", content="Q2"),
            ],
            metadata={"policy_id": "prbench_v1"},
        )
        msgs = build_candidate_messages(ex, "DEFAULT_SYS")
        self.assertEqual([(m.role, m.content) for m in msgs], [("user", "Q1"), ("assistant", "A1"), ("user", "Q2")])

    def test_apex_policy_uses_explicit_system_and_user_template(self) -> None:
        ex = NormalizedExample(
            id="ax-1",
            dataset_name="d",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="Prompt",
            messages=[],
            metadata={
                "policy_id": "apexv1_extended_v1",
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
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[0].content, APEX_V1_GENERATION_SYSTEM_PROMPT)
        self.assertEqual(msgs[1].role, "user")
        user_text = msgs[1].content
        self.assertIn("Task Domain: Legal", user_text)
        self.assertIn("Task Prompt: Prompt", user_text)
        self.assertIn("Attachments:", user_text)
        self.assertIn("==== Attached files content", user_text)
        self.assertIn("documents/1382/file.pdf (pdf)", user_text)
        self.assertIn("Convention body text", user_text)

    def test_apex_policy_ignores_default_system_prompt(self) -> None:
        ex = NormalizedExample(
            id="ax-2",
            dataset_name="d",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="Prompt 2",
            messages=[LLMMessage(role="user", content="ignored")],
            metadata={
                "policy_id": "apexv1_extended_v1",
                "attachment_contents": [
                    {"path": "documents/1382/file.pdf", "kind": "pdf", "text": "Body"}
                ],
            },
        )
        msgs = build_candidate_messages(ex, "DEFAULT_SYS")
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[0].content, APEX_V1_GENERATION_SYSTEM_PROMPT)
        self.assertNotIn("DEFAULT_SYS", msgs[0].content)
        self.assertEqual(msgs[1].role, "user")
        self.assertIn("Task Prompt: Prompt 2", msgs[1].content)

    def test_lexam_open_policy_instructions_present(self) -> None:
        expected_template = """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
Assume Swiss law applies unless specifically mentioned; if the course context justifies, address legal issues beyond Swiss law as well.
Use precise legal language and formal "Sie" when answering.
Do NOT state any disclaimer or refer to the need for external legal advice.
Do NOT request the user to consult laws or to research on their own.
Offer focused legal analyses and individualized advice.
Speak directly and authoritatively without mentioning that your response is merely for general information.
Incorporate Swiss-specific legal terminology.
If you have discovered relevant legal considerations (Erwägungen), respond with a concise, clear legal analysis.
Cite only from your identified considerations.
Always cite the specific legal provision, explicitly indicating paragraphs (Abs.), numbers (Ziff.), or letters (lit.) where available (e.g., “'Art. 74 Abs. 2 Ziff. 2 OR”, “Art. 336 lit. a StGB”). Avoid general references (such as 'Art. 3 ZGB') without mentioning the specific paragraph, number, or letter, if applicable.
If no relevant considerations are found, explicitly state that no pertinent information is available.
If you do have reliable sources, share practical guidance or insights from them.
Respond in the same language as the question.
If the question specifically requests a short answer, provide a concise response.
If the prompt asks you to analyze a specific case provided in the exam, but the text or details of that case have not been provided in the prompt, explicitly flag that the required case material is missing.

Question:
{question}

Answer:"""
        self.assertEqual(LEXAM_QA_PROMPT, expected_template)

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
        self.assertFalse(any(m.role == "system" for m in msgs))
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(
            msgs[0].content,
            LEXAM_QA_PROMPT.format(course_name="Swiss Law", question="Frage"),
        )
        self.assertIn('formal "Sie"', msgs[0].content)
        self.assertIn("Erwägungen", msgs[0].content)
        self.assertIn("Art. 74 Abs. 2 Ziff. 2 OR", msgs[0].content)

    def test_lexam_mcq_prompt_matches_lexam_structure_json_output(self) -> None:
        expected_template = """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
You are given a multiple-choice question, where only one choice (e.g., A, B, C, etc.) is correct.
Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, consider legal frameworks beyond Swiss law as well.

Please reason through the question step by step, using a chain-of-thought approach:
- Clarify the facts: Briefly restate or highlight the key facts in the question to anchor your reasoning.
- Issue Identification: What legal issue(s) arise from the facts?
- Rule Explanation: What legal rules or principles are relevant, and what are their sources (e.g., statutes, case law, doctrine)?
- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing any ambiguities, exceptions, or competing interpretations.
- Eliminate Incorrect Answers: Briefly explain why each incorrect answer is wrong or less convincing.
- Conclusion: Clearly state the correct answer choice (e.g., A, B, C, etc.) with a brief justification for why it best fits the legal analysis.

Question:
 {question}

Answer:"""
        self.assertEqual(LEXAM_MCQ_PROMPT_LETTERS, expected_template)

        ex = NormalizedExample(
            id="lx-mcq-1",
            dataset_name="d",
            provenance="canonical:mcq",
            judge_mode="mcq",
            instructions="Welche Norm gilt?\n\nA. Erste\nB. Zweite",
            messages=[],
            metadata={"policy_id": "lexam_mcq_v1", "course": "Privatrecht"},
        )
        msgs = build_candidate_messages(ex, "DEFAULT_SYS")
        self.assertFalse(any(m.role == "system" for m in msgs))
        self.assertEqual(len(msgs), 1)
        user_text = msgs[0].content
        self.assertIn("Please reason through the question step by step", user_text)
        self.assertIn('{"answer": "<choice_id>", "reasoning": "<short text>"}', user_text)
        self.assertIn("A. Erste", user_text)
        self.assertIn("B. Zweite", user_text)

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

    def test_scratchpad_is_not_included_by_default(self) -> None:
        ex = NormalizedExample(
            id="pr-2",
            dataset_name="d",
            provenance="canonical:rubric_qa",
            judge_mode="rubric",
            instructions="Prompt",
            messages=[],
            metadata={"policy_id": "prbench_v1", "scratchpad": "private notes"},
        )
        msgs = build_candidate_messages(ex, "DEFAULT_SYS")
        all_text = "\n".join(m.content for m in msgs)
        self.assertNotIn("private notes", all_text)

