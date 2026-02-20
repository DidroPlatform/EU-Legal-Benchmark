from __future__ import annotations

from typing import List

from ..data.policies import get_policy
from ..types import LLMMessage, NormalizedExample

APEX_V1_GENERATION_SYSTEM_PROMPT = """SYSTEM_PROMPT
You are an AI assistant that produces final, domain-appropriate deliverables from a given task description and (optionally) attached files. You will be given the following inputs:
Inputs
• Task Domain: <Domain>  (e.g., "Operations")
• Task Prompt: <Prompt>
• Attachments:
  ==== Attached files content: ====
  === <File_1> ===
  <File_1_Contents>
  === <File_2> ===
  <File_2_Contents>
  … (repeat as needed)
Ground Rules
1) You must not ask follow-up questions. Interpret the prompt as best you can and produce the best complete answer given the information provided.
2) Use the attachments as primary sources.
3) Treat each "=== <File_Name> === …" block as the full content of that file.
All of the source files that you need have been added to the prompt."""

APEX_V1_GENERATION_USER_PROMPT_TEMPLATE = """Inputs
• Task Domain: Legal
• Task Prompt: {task_prompt}
• Attachments:
"""

LEXAM_QA_PROMPT = """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
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

LEXAM_MCQ_PROMPT_LETTERS = """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
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


_MCQ_JSON_OUTPUT_INSTRUCTION = (
    "Return only valid JSON with no markdown. "
    'Use this schema exactly: {"answer": "<choice_id>", "reasoning": "<short text>"}. '
    "The answer must be exactly one of the provided choice IDs."
)


def _build_policy_guidance(example: NormalizedExample) -> str:
    policy = get_policy(example.metadata.get("policy_id"))
    parts: List[str] = []

    if policy.generation_prefix:
        parts.append(policy.generation_prefix)

    return "\n".join([p for p in parts if p.strip()])


def _render_attachment_content(example: NormalizedExample) -> str:
    contents = example.metadata.get("attachment_contents")
    if isinstance(contents, list) and contents:
        blocks = []
        for item in contents:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            kind = str(item.get("kind", "")).strip()
            text = str(item.get("text", "")).strip()
            error = str(item.get("error", "")).strip()

            label = f"{path} ({kind})" if kind else path
            if text:
                blocks.append(f"=== {label} ===\n{text}")
            elif error:
                blocks.append(f"=== {label} ===\n[Parsing error] {error}")
            else:
                blocks.append(f"=== {label} ===\n[No extractable text]")
        if blocks:
            return "==== Attached files content: ====\n\n" + "\n\n".join(blocks)

    attachments = example.metadata.get("attachments")
    if isinstance(attachments, list) and attachments:
        attachment_lines = []
        for item in attachments:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            kind = str(item.get("kind", "")).strip()
            if kind:
                attachment_lines.append(f"- {path} ({kind})")
            else:
                attachment_lines.append(f"- {path}")
        if attachment_lines:
            return "==== Attached files content: ====\n" + "\n".join(attachment_lines)
    return ""

def build_candidate_messages(example: NormalizedExample, system_prompt: str) -> List[LLMMessage]:
    policy = get_policy(example.metadata.get("policy_id"))
    policy_guidance = _build_policy_guidance(example)

    if policy.policy_id == "lexam_oq_v1":
        course_name = str(example.metadata.get("course") or "Swiss Law").strip()
        return [
            LLMMessage(
                role="user",
                content=LEXAM_QA_PROMPT.format(
                    course_name=course_name,
                    question=example.instructions.strip(),
                ),
            )
        ]

    if policy.policy_id == "lexam_mcq_v1":
        course_name = str(example.metadata.get("course") or "Swiss Law").strip()
        prompt = LEXAM_MCQ_PROMPT_LETTERS.format(
            course_name=course_name,
            question=example.instructions.strip(),
        )
        return [
            LLMMessage(
                role="user",
                content=f"{prompt}\n\n{_MCQ_JSON_OUTPUT_INSTRUCTION}",
            )
        ]

    if policy.policy_id == "apexv1_extended_v1":
        rendered_attachments = _render_attachment_content(example)
        user_parts = [
            APEX_V1_GENERATION_USER_PROMPT_TEMPLATE.format(
                task_prompt=example.instructions.strip()
            ).rstrip()
        ]
        if rendered_attachments:
            user_parts.append(rendered_attachments)

        return [
            LLMMessage(role="system", content=APEX_V1_GENERATION_SYSTEM_PROMPT),
            LLMMessage(role="user", content="\n".join(user_parts)),
        ]

    system_messages: List[LLMMessage] = []
    if policy.use_default_system_prompt:
        system_messages.append(LLMMessage(role="system", content=system_prompt))

    if example.messages:
        base = [*system_messages, *example.messages]
        if policy_guidance:
            base.append(LLMMessage(role="user", content=policy_guidance))
        if example.judge_mode == "mcq" and policy.mcq_json_answer:
            base.append(LLMMessage(role="user", content=_MCQ_JSON_OUTPUT_INSTRUCTION))
        return base

    parts = [example.instructions.strip()]
    if example.context.strip():
        parts.append("Context:\n" + example.context.strip())
    if policy_guidance:
        parts.append(policy_guidance)
    if example.judge_mode == "mcq" and policy.mcq_json_answer:
        parts.append(_MCQ_JSON_OUTPUT_INSTRUCTION)

    return [*system_messages, LLMMessage(role="user", content="\n\n".join([p for p in parts if p]))]
