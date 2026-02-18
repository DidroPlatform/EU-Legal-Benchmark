# Build Minimal Benchmark Eval Scaffold

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` are updated during implementation.

This plan follows repository guidance in `.agent/PLANS.md`.

## Purpose / Big Picture

After this change, a user can run `python run.py --config config.example.yaml` and execute an end-to-end benchmark pipeline: load dataset examples, generate model responses with multiple providers, judge responses with an LLM-as-a-judge model, and write stable output artifacts (`responses.jsonl`, `judgments.jsonl`, `summary.json`).

The scaffold is intentionally minimal while still complete: typed request/response contracts, provider adapters, disk cache, retry/backoff, deterministic request IDs, and config-driven execution.

## Progress

- [x] (2026-02-08 00:00Z) Reviewed constraints, dataset files in `data/`, and PRBench patterns.
- [x] (2026-02-08 00:00Z) Drafted architecture and file layout for a minimal, extensible scaffold.
- [x] (2026-02-08 00:00Z) Implemented provider abstraction and provider clients (OpenAI, Vertex, Mistral).
- [x] (2026-02-08 00:00Z) Implemented data normalization, prompt building, judge prompt + parsing, and run CLI.
- [x] (2026-02-08 00:00Z) Added config example, README, and dependency updates.
- [x] (2026-02-08 00:00Z) Run a smoke test (without live API calls) to verify module imports and CLI help.
- [x] (2026-02-15 00:00Z) Consolidated runtime provider calls through LiteLLM and removed legacy native provider adapter modules.
- [x] (2026-02-15 00:00Z) Refactored rubric judging flow to execute one independent LLM call per rubric criterion.
- [x] (2026-02-15 00:00Z) Added `judges` config list support with legacy `judge` backward compatibility and round-robin criterion assignment.
- [x] (2026-02-15 00:00Z) Added parallel rubric judging with configurable Gemini-safe rate limiting (`run.judge_parallel_workers`, `run.judge_rate_limit_rpm`).
- [ ] Validate end-to-end run artifact shape for multi-judge rubric metadata on a live or mocked run.

## Surprises & Discoveries

- Observation: `data/` contains multiple JSONL schemas (PRBench-like, legalqaeval-like, includebase-like).
  Evidence: key inspection across files showed different field sets and label formats.
- Observation: PRBench’s public harness uses config-driven execution, disk cache, retries, and stable output files, but is tailored to an OpenAI-compatible endpoint.
  Evidence: `scaleapi/PRBench` files `evals.py`, `util.py`, `config.py`, `README.md`.

## Decision Log

- Decision: Initial plan explored thin native provider clients; this was later superseded by LiteLLM-only routing.
  Rationale: Keeping this record preserves implementation history and explains why the final design consolidated to one runtime adapter path.
  Date/Author: 2026-02-08 (superseded 2026-02-15) / Codex

- Decision: Use synchronous execution for minimal complete flow.
  Rationale: Simpler control flow and easier debugging for an initial scaffold; abstractions keep async/concurrency as a future additive change.
  Date/Author: 2026-02-08 / Codex

- Decision: Consolidate runtime provider calls to LiteLLM and retire native provider adapters.
  Rationale: One adapter path reduces setup complexity and dependency sprawl while preserving multi-provider model support via LiteLLM routing.
  Date/Author: 2026-02-15 / Codex

- Decision: Normalize data into one in-memory schema and branch loader logic by observed keys.
  Rationale: Existing datasets already vary in shape; normalization prevents dataset-specific logic from leaking into providers and judge code.
  Date/Author: 2026-02-08 / Codex

- Decision: Execute rubric grading as one call per criterion and aggregate scores deterministically.
  Rationale: This enforces independence across criteria, enables heterogeneous judge models per criterion, and avoids single-call cross-criterion leakage.
  Date/Author: 2026-02-15 / Codex

- Decision: Assign rubric criteria to configured judges by deterministic round-robin over rubric order.
  Rationale: Round-robin is simple, repeatable, and supports both equal-size and unequal-size rubric/judge pools without extra config surface.
  Date/Author: 2026-02-15 / Codex

- Decision: Parallelize rubric criterion judge calls and apply a shared per-minute throttle when Gemini judge providers are used.
  Rationale: Criterion calls are independent and parallelizable; a conservative throttle reduces 429 retries while still accelerating grading.
  Date/Author: 2026-02-15 / Codex

## Outcomes & Retrospective

Implementation completed for the requested minimal scaffold and artifact pipeline, then refactored to support multi-provenance datasets in one run and to keep repository root layout clean. The structure is extensible for new providers, datasets, and judge strategies. Residual operational work is mostly credentials and optional tuning (concurrency, stricter JSON schema validation, richer analytics).

Additional enhancement: rubric grading now performs independent per-criterion calls and deterministic weighted aggregation using criterion-level scores, with a configurable judge pool (`judges`) and legacy single-judge fallback (`judge`).

## Context and Orientation

Key modules:

- `run.py` orchestrates config loading, data loading, generation, judging, aggregation, and output writes.
- `src/providers/litellm.py` implements runtime model invocation for all providers via LiteLLM behind `src/providers/base.py`.
- `src/data/loader.py` reads JSONL files and normalizes examples across provenances.
- `src/prompting/templates.py` creates candidate prompt messages.
- `src/judge/judge.py` builds judge prompts and parses strict judge JSON outputs.
- `src/cache.py` provides disk caching keyed by request payload hash.
- `src/retry.py` provides retry/backoff with transient error detection.

## Plan of Work

Implement typed core request/response models and a provider interface first. Then implement a LiteLLM-based provider adapter that maps the shared request into provider-routed calls and normalizes returned usage and text. Add a dataset loader that detects known row schemas and maps each row to a single `NormalizedExample` type. Add prompt builders for candidate generation and judge requests. Build a run CLI that loops through candidate models and examples, uses cache+retry wrappers for generation and judging, writes JSONL artifacts, and emits summary metrics.

Finally, add `config.example.yaml`, update dependencies, and document commands and dataset assumptions in `README.md`.

## Concrete Steps

From repository root:

    python run.py --help

Then with credentials configured:

    python run.py --config config.example.yaml

Expected artifact layout:

    outputs/<run_id>/examples.jsonl
    outputs/<run_id>/responses.jsonl
    outputs/<run_id>/judgments.jsonl
    outputs/<run_id>/trace.jsonl
    outputs/<run_id>/summary.json

## Validation and Acceptance

Acceptance criteria:

- CLI runs with `python run.py --config ...`.
- At least one candidate model response is produced and judged.
- Output files are written under `outputs/<run_id>/`.
- `summary.json` contains per-model counts and average score/pass-rate.
- Cache hits occur on reruns for unchanged requests.

## Idempotence and Recovery

The run is idempotent with caching enabled: repeated runs with same config and messages will reuse cached responses. If a provider call fails transiently, retry/backoff is applied. If a run fails mid-way, re-running with the same config and run ID can reuse cache and regenerate artifacts safely.

## Artifacts and Notes

Reference patterns mirrored from PRBench:

- Config-driven entry point and model lists.
- Disk cache keys from model + prompt + decoding params.
- Retry with backoff on transient failures.
- Structured result outputs and run-level summary.

## Interfaces and Dependencies

Core interface contract:

- `src.types.LLMRequest`: shared request schema for all providers.
- `src.types.LLMResponse`: shared response schema from all providers.
- `src.providers.base.BaseProvider.generate(request) -> LLMResponse`.

Runtime dependencies:

- `litellm` (single runtime adapter for provider-routed model calls)
- `PyYAML` (config parsing)

Update note (2026-02-08, Codex): initial implementation completed, then refactored for multi-dataset provenance routing, full trace artifacts, and clean root layout (`run.py` + docs/config, internal code in `src/`).
Update note (2026-02-15, Codex): updated plan context to reflect LiteLLM-only runtime adapter after removing native provider modules and simplifying dependencies.
Update note (2026-02-15, Codex): updated execution flow and config model for per-criterion rubric judging with independent calls and multi-judge assignment.
Update note (2026-02-15, Codex): cleaned wording for current repo paths and clarified that early native-provider notes are historical and superseded.
Update note (2026-02-15, Codex): added bounded parallel rubric judging and Gemini-focused judge RPM throttling.
