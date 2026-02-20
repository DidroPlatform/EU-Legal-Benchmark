from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetPolicy:
    policy_id: str
    use_default_system_prompt: bool = True
    generation_prefix: str = ""
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
    # APEX v1 extended: custom generation/judging templates are handled in prompting/judge modules.
    "apexv1_extended_v1": DatasetPolicy(
        policy_id="apexv1_extended_v1",
        use_default_system_prompt=False,
        mcq_json_answer=True,
        rubric_judge_style="default",
    ),
    # LEXam open questions: custom generation template handled in prompting module.
    "lexam_oq_v1": DatasetPolicy(
        policy_id="lexam_oq_v1",
        use_default_system_prompt=False,
        mcq_json_answer=True,
        rubric_judge_style="default",
    ),
    # LEXam MCQ path is metric-based and non-judge in source benchmark.
    "lexam_mcq_v1": DatasetPolicy(
        policy_id="lexam_mcq_v1",
        use_default_system_prompt=False,
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
