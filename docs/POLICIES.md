# Dataset Policies

This file defines how canonical rows (`legal_eval_v1`) are mapped to dataset-specific
prompting and judging behavior at runtime.

The runtime policy key is stored in `metadata.policy_id` on each row.

## Active policy mapping

- `prbench_v1`
  - Source dataset: `prbench`
  - Generation: use dataset conversation turns from canonical `messages`; do **not** inject default system prompt.
  - Generation: all turn-indexed `*reference*_<i>` list columns are prepended into matching user turns during build.
  - Generation: `scratchpad` is metadata-only by default (not sent to candidate model).
  - Rubric judging: criterion-first style using PRBench GRADER_TEMPLATE-compatible prompt with
    `<<conversation>>` + `<<rubric_item>>` substitution.
  - Rubric judging: response text is cleaned for common reasoning tags (`<think>`, `<reasoning>`, etc.) before transcript scoring.

- `apexv1_extended_v1`
  - Source dataset: `apexv1`
  - Generation: use benchmark-style explicit APEX system prompt (`SYSTEM_PROMPT` block)
    and a single user prompt scaffold:
    `Inputs`, `Task Domain: Legal`, `Task Prompt: <Prompt>`, `Attachments:`.
  - Generation: when attachments exist, parse local files (PDF/text) and inject explicit
    `==== Attached files content: ====`
    block with extracted attachment text under `Attachments:`.
  - Rubric judging: criterion-level APEX grading template with JSON output
    `{"result": <1 or 0>, "reason": "<concise explanation>"}`.

- `lexam_oq_v1`
  - Source dataset: `lexam` open questions (`task_type=reference_qa`)
  - Generation: use explicit LEXam `QA_PROMPT` template (from upstream LEXam source),
    injected as a single `user` message.
  - Generation: do not inject the benchmark default system prompt.
  - Judging: for `judge_mode=reference`, use explicit LEXam judge system/user prompt
    content (from upstream LEXam source) with this benchmark's JSON judge output schema.
  - Scoring: normalize judge score to `0.0..1.0` with `0.1` increments using round-half-up.

- `lexam_mcq_v1`
  - Source dataset: `lexam` MCQ (`task_type=mcq`)
  - Generation: use explicit LEXam MCQ chain-of-thought template (letters variant) as a single
    `user` message; append benchmark JSON answer contract for deterministic parsing.
  - Generation: do not inject the benchmark default system prompt.
  - Loader shaping: compose MCQ instructions as `question + option lines` only (no `Choices:`
    header and no generic trailing instruction).
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
The fallback is deterministic and non-fatal: run execution continues with baseline prompting/judging defaults.
