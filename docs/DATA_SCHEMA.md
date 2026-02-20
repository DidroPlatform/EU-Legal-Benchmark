# Canonical Dataset Schema (`legal_eval_v1`)

This document defines the on-disk JSONL contract for benchmark input data.
Each line must be one JSON object following this schema.

## Purpose

Use one canonical input format across datasets so the benchmark pipeline can run
with a single ingestion and grading flow.

## Top-level fields

Required for all rows:

- `schema_version` (string): must be `"legal_eval_v1"`.
- `id` (string): globally unique and stable ID.
- `dataset` (string): source dataset name (`prbench`, `apexv1`, `lexam`, etc.).
- `task_type` (string): one of `rubric_qa`, `reference_qa`, `mcq`.
- `prompt` (string): user-facing question/instruction.

Optional for all rows:

- `context` (string): additional supporting text. If absent, treated as `""`.
- `messages` (array): optional conversation turns, each object:
  - `role` (string, required): one of `user`, `assistant`, `system`.
  - `content` (string, required): non-empty message text.
- `attachments` (array): each item is an object with:
  - `path` (string, required): relative path to local artifact/document.
  - `kind` (string, optional): attachment type hint (`pdf`, `image`, etc.).
  - `title` (string, optional): human-readable label.
- `metadata` (object): free-form provenance and dataset-specific attributes.
  - Recommended key: `metadata.policy_id` (string), used to select dataset-specific prompting/judging behavior.

## Task types

### `rubric_qa`

Required:

- `rubric` (non-empty array of objects)

Optional:

- `reference_answers` (array of strings)

Forbidden:

- `choices`
- `correct_choice_ids`

Rubric criterion object shape:

- `id` (string, required)
- `title` (string, required)
- `description` (string, optional)
- `weight` (number, optional; if absent, consumers may assume `1.0`)

### `reference_qa`

Required:

- `reference_answers` (non-empty array of non-empty strings)

Forbidden:

- `rubric`
- `choices`
- `correct_choice_ids`

### `mcq`

Required:

- `choices` (array with at least 2 objects)
- `correct_choice_ids` (non-empty array of strings, each must exist in `choices[*].id`)

Forbidden:

- `rubric`
- `reference_answers`

Choice object shape:

- `id` (string, required; recommended labels `A`, `B`, `C`, `D`)
- `text` (string, required)

## Validation policy

Validation is strict for required/forbidden fields and field types.
Unknown top-level fields are allowed but should be moved into `metadata` over time.

When `messages` is provided, it must be well-formed (`role` + non-empty `content`),
or row validation fails.

Validation ownership note:
- Config-time checks (for example response-source mode validity and required config paths) live in `src/config.py`.
- Runtime checks (for example selected-pair coverage and dynamic row-content constraints) execute in runner flow (`src/runner/orchestrator.py`, `src/runner/generation.py`).

## ID stability rule

`id` must remain stable across re-runs of the same converter and input row.
Use deterministic IDs derived from immutable source identifiers (or a stable hash if needed).

## Grading contract

- `rubric_qa`: graded with LLM judge against `rubric` (and optionally references/context).
- `reference_qa`: graded with LLM judge against `reference_answers` and optional `context`.
- `mcq`: graded programmatically using `correct_choice_ids` (no LLM judge needed).
