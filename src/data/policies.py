from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetPolicy:
    policy_id: str
    use_default_system_prompt: bool = True
    generation_prefix: str = ""
    require_same_language: bool = False
    citation_style_hint: str = ""
    handle_missing_material: bool = False
    include_domain_header: bool = False
    include_attachment_block: bool = False
    mcq_json_answer: bool = True
    rubric_judge_style: str = "default"


POLICIES: dict[str, DatasetPolicy] = {
    # PRBench: conversation prompts from dataset turns, no extra default system prompt in original harness.
    "prbench_v1": DatasetPolicy(
        policy_id="prbench_v1",
        use_default_system_prompt=False,
        generation_prefix="",
        mcq_json_answer=True,
        rubric_judge_style="criterion_binary",
    ),
    # APEX v1 extended: prompt is template-based user content with rubric grading template.
    "apexv1_extended_v1": DatasetPolicy(
        policy_id="apexv1_extended_v1",
        use_default_system_prompt=False,
        generation_prefix=(
            "You are solving an APEX-style legal evaluation prompt. "
            "Use only the provided task statement and attached materials."
        ),
        include_domain_header=True,
        include_attachment_block=True,
        mcq_json_answer=True,
        rubric_judge_style="default",
    ),
    # LEXam open questions: formal Swiss-law style response expectation.
    "lexam_oq_v1": DatasetPolicy(
        policy_id="lexam_oq_v1",
        use_default_system_prompt=True,
        generation_prefix=(
            "You are a Swiss-law exam expert. Answer in formal legal style."
        ),
        require_same_language=True,
        citation_style_hint="Use provision-level citation format with Abs., Ziff., lit. where applicable.",
        handle_missing_material=True,
        mcq_json_answer=True,
        rubric_judge_style="default",
    ),
    # LEXam MCQ path is metric-based and non-judge in source benchmark.
    "lexam_mcq_v1": DatasetPolicy(
        policy_id="lexam_mcq_v1",
        use_default_system_prompt=True,
        generation_prefix="Select the single best choice from the provided options.",
        mcq_json_answer=True,
        rubric_judge_style="default",
    ),
    # Include-base has no custom benchmark-specific prompting rules documented yet.
    "includebase_default_v1": DatasetPolicy(
        policy_id="includebase_default_v1",
        use_default_system_prompt=True,
        generation_prefix="",
        mcq_json_answer=True,
        rubric_judge_style="default",
    ),
    # LAR-ECHR MCQ: deterministic option selection from facts + argument continuation.
    "lar_echr_mcq_v1": DatasetPolicy(
        policy_id="lar_echr_mcq_v1",
        use_default_system_prompt=True,
        generation_prefix=(
            "You are answering an ECHR argument-continuation multiple-choice question. "
            "Choose the single best continuation based on the provided facts and argument excerpt."
        ),
        mcq_json_answer=True,
        rubric_judge_style="default",
    ),
}


DEFAULT_POLICY = DatasetPolicy(policy_id="default_v1")


def get_policy(policy_id: str | None) -> DatasetPolicy:
    if not policy_id:
        return DEFAULT_POLICY
    return POLICIES.get(policy_id, DEFAULT_POLICY)
