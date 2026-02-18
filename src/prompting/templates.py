from __future__ import annotations

from typing import List

from ..data.policies import get_policy
from ..types import LLMMessage, NormalizedExample


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

    if policy.include_domain_header:
        domain = str(example.metadata.get("domain", "")).strip()
        if domain:
            parts.append(f"Domain: {domain}")

    if policy.require_same_language:
        parts.append(
            "Write your final answer in the same language as the question unless explicitly asked otherwise."
        )

    if policy.citation_style_hint:
        parts.append(policy.citation_style_hint)

    if policy.handle_missing_material:
        parts.append(
            "If key factual material is missing, state what is missing explicitly before giving your best qualified answer."
        )

    if policy.include_attachment_block:
        rendered = _render_attachment_content(example)
        if rendered:
            parts.append(rendered)

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
                blocks.append(f"File: {label}\n{text}")
            elif error:
                blocks.append(f"File: {label}\n[Parsing error] {error}")
            else:
                blocks.append(f"File: {label}\n[No extractable text]")
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


def _should_merge_policy_into_first_user_message(example: NormalizedExample) -> bool:
    return str(example.metadata.get("policy_id", "")).strip() == "apexv1_extended_v1"


def build_candidate_messages(example: NormalizedExample, system_prompt: str) -> List[LLMMessage]:
    policy = get_policy(example.metadata.get("policy_id"))
    policy_guidance = _build_policy_guidance(example)

    system_messages: List[LLMMessage] = []
    if policy.use_default_system_prompt:
        system_messages.append(LLMMessage(role="system", content=system_prompt))

    if example.messages:
        base = [*system_messages, *example.messages]
        if policy_guidance:
            if _should_merge_policy_into_first_user_message(example):
                merged = False
                for idx, msg in enumerate(base):
                    if msg.role == "user":
                        merged_content = msg.content.rstrip() + "\n\n" + policy_guidance
                        base[idx] = LLMMessage(role="user", content=merged_content)
                        merged = True
                        break
                if not merged:
                    base.append(LLMMessage(role="user", content=policy_guidance))
            else:
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
