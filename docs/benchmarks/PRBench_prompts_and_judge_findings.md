# PRBench Prompt & Judge Findings

This document summarizes which prompts are used in PRBench for:
- Question prompts (conversation input to the response model)
- Judge prompts (rubric grading)

Scope note: this document is based on the upstream PRBench codebase and references external PRBench paths, not local files in this repository.

## 1. Question Prompts

### Source of question prompts
- PRBench does **not** define a single hardcoded "question template" for task prompts.
- It loads conversation turns directly from the `ScaleAI/PRBench` dataset:
  - User turns: `prompt_0` ... `prompt_9`
  - Assistant turns: `response_0` ... `response_9`
- Relevant code:
  - `PRBench/evals.py:19`
  - `PRBench/evals.py:36`
  - `PRBench/evals.py:38`
  - `PRBench/evals.py:39`
  - `PRBench/evals.py:41`

### Reference text prepending
- Before running evals, PRBench checks `reference_texts_i` columns.
- If a reference list is non-empty, it prepends:
  - `Reference Text 0: ...`
  - `Reference Text 1: ...`
  - etc.
  to the corresponding `prompt_i`.
- Relevant code:
  - `PRBench/evals.py:29`
  - `PRBench/util.py:318`
  - `PRBench/util.py:327`
  - `PRBench/util.py:328`

### Prompt shape sent to response model
- For sampled responses, the harness sends conversation messages as:
  - `[{ "role": role, "content": content }, ...]`
- There is no separate default system prompt injected for generation in this flow.
- Relevant code:
  - `PRBench/util.py:183`
  - `PRBench/util.py:185`
  - `PRBench/util.py:189`

## 2. Judge Prompt

### Main grading template
- Judge prompt template is `GRADER_TEMPLATE` in:
  - `PRBench/constants.py:1`
- It includes placeholders:
  - `<<conversation>>`
  - `<<rubric_item>>`

### How it is instantiated
- For each rubric criterion:
  1. Replace `<<rubric_item>>` with criterion text (rubric title).
  2. Replace `<<conversation>>` with full conversation string (including final candidate assistant response).
- Relevant code:
  - `PRBench/util.py:160`
  - `PRBench/util.py:163`
  - `PRBench/util.py:164`
  - `PRBench/util.py:224`
  - `PRBench/evals.py:53`
  - `PRBench/evals.py:63`

### Expected judge output
- The template asks judge to return markdown JSON containing:
  - `explanation` (string)
  - `criteria_met` (boolean)
- Relevant code:
  - `PRBench/constants.py:10`
  - `PRBench/constants.py:11`
  - `PRBench/constants.py:12`
  - `PRBench/constants.py:48`

### Score extraction and aggregation
- Parser extracts `criteria_met` from judge text and maps:
  - `"true" -> 1`
  - `"false" -> 0`
- Weighted sum is then computed over rubric weights.
- Relevant code:
  - `PRBench/util.py:200`
  - `PRBench/util.py:210`
  - `PRBench/constants.py:51`
  - `PRBench/util.py:179`

## 3. Judge Model Configuration

- Judge model is set via config:
  - `judge_model_name` in `PRBench/config.yaml`
- Repository default config uses:
  - `openai/o4-mini`
- Relevant code:
  - `PRBench/config.yaml:1`
  - `PRBench/evals.py:121`

## 4. Additional Templates Present but Not in Main Eval Flow

- `PRBench/constants.py` includes:
  - `LEGAL_DECISION_TYPE_TEMPLATE`
  - `LEGAL_ECONOMIC_PATHWAY_TEMPLATE`
  - `FINANCE_DECISION_TYPE_TEMPLATE`
  - `FINANCE_ECONOMIC_PATHWAY_TEMPLATE`
- In current `evals.py` flow, these are not wired into grading.
- Relevant references:
  - `PRBench/constants.py:57`
  - `PRBench/constants.py:123`
  - `PRBench/constants.py:158`
  - `PRBench/constants.py:218`
