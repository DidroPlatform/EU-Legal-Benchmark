# Dataset Policies

This file defines how canonical rows (`legal_eval_v1`) are mapped to dataset-specific
prompting and judging behavior at runtime.

The runtime policy key is stored in `metadata.policy_id` on each row.

## Active policy mapping

- `prbench_v1`
  - Source dataset: `prbench`
  - Generation: use dataset conversation turns; do **not** inject default system prompt.
  - Rubric judging: criterion-first style with binary criterion guidance.

- `apexv1_extended_v1`
  - Source dataset: `apexv1`
  - Generation: no default system prompt; user-side prompt includes APEX-style task framing.
  - Generation: inject `Domain: ...` header when available.
  - Generation: when attachments exist, parse local files (PDF/text) and inject explicit
    `==== Attached files content: ====`
    block with extracted attachment text.
  - Generation: for strict APEX mimic, merge policy guidance into the first user message
    instead of sending it as a separate follow-up user turn.
  - Rubric judging: default rubric style.

- `lexam_oq_v1`
  - Source dataset: `lexam` open questions (`task_type=reference_qa`)
  - Generation: keep default system prompt.
  - Generation: enforce Swiss-law exam style guidance, same-language response, citation hint
    (`Abs.`, `Ziff.`, `lit.`), and explicit handling for missing case material.
  - Judging: default reference-qa style.

- `lexam_mcq_v1`
  - Source dataset: `lexam` MCQ (`task_type=mcq`)
  - Generation: keep default system prompt; enforce JSON answer format for deterministic grading.
  - Grading: programmatic exact-match (no LLM judge).

- `includebase_default_v1`
  - Source dataset: `includebase` (`task_type=mcq`)
  - Generation: default behavior.
  - Grading: programmatic exact-match (no LLM judge).

- `lar_echr_mcq_v1`
  - Source dataset: `lar_echr` (`task_type=mcq`)
  - Generation: keep default system prompt and add short ECHR continuation-task guidance.
  - Grading: programmatic exact-match (no LLM judge).

## Fallback

If `metadata.policy_id` is missing or unknown, runtime falls back to default behavior.
