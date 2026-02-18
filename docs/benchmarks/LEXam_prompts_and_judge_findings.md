# LEXam Prompt & Judge Findings

## Scope
This document summarizes which prompts are used in LEXam for:
- Question answering (open questions and MCQs)
- LLM-as-judge grading

It also notes active defaults and implementation quirks.
Scope note: references in this file point to the upstream LEXam benchmark codebase, not files in this repository.

## 1. Question Prompts

### Open Questions (`QA_PROMPT`)
Defined in:
- `LEXam/lighteval/community_tasks/lexam_oq_evals.py:75`
- `LEXam/litellm_eval.py:13` (duplicate for direct LiteLLM path)

Behavior:
- Frames the model as a Swiss-law exam expert.
- Enforces formal/legal style, Swiss terminology, and provision-level citation detail (`Abs.`, `Ziff.`, `lit.`).
- Requires same-language response and handling of missing case material.

Usage path:
- Lighteval task prompt function: `LEXam/lighteval/community_tasks/lexam_oq_evals.py:247`
- Direct LiteLLM path (`task_type=open_questions`): `LEXam/litellm_eval.py:332`

### MCQ Prompts (`MCQ_PROMPT`)
Defined in:
- `LEXam/lighteval/community_tasks/lexam_mcq_evals.py:79`
- `LEXam/litellm_eval.py:36` (duplicate for direct LiteLLM path)

Variants:
- `letters` expects final format like `Correct Answer: ###C###`
- `numbers` expects final format like `Correct Answer: ###3###`

Active default in lighteval MCQ task:
- `MCQ_PROMPT_KEY = "letters"` at `LEXam/lighteval/community_tasks/lexam_mcq_evals.py:63`

Usage path:
- Lighteval MCQ prompt function: `LEXam/lighteval/community_tasks/lexam_mcq_evals.py:121`
- Direct LiteLLM path:
  - `mcq_letters`: `LEXam/litellm_eval.py:310`
  - `mcq_numbers`: `LEXam/litellm_eval.py:321`

## 2. Judge Prompts (Open Questions)

## Lighteval judge stack (main benchmark path)
Defined in `LEXam/lighteval/community_tasks/lexam_oq_evals.py`:
- Prompt key default: `JUDGE_PROMPT_KEY = "20250324"` (`:60`)
- System prompt variants: `JUDGE_SYSTEM` (`:97`)
- User prompt variants: `JUDGE_USER` (`:103`)
- Dynamic instruction wrapper with Question/Reference/Model answer: `:211`

How composed:
1. Select `system_prompt = JUDGE_SYSTEM[system_style]` (`:208`)
2. Select `user = JUDGE_USER[system_style]` (`:209`)
3. Append instruction block with:
   - `Question`
   - `Reference Answer`
   - `Model's Answer`
   (`:211` to `:224`)
4. Return chat messages `[system, user]` (`:228`)

Active defaults:
- Judge style key: `20250324` (`:60`)
- Metrics mode: `metrics_to_evaluate = ["judge"]` (`:283`)
- Judge model default (with `USE_MINI = False`): `openai/gpt-4o-2024-11-20` (`:271`)

## Direct grading script (`customized_judge_async.py`)
Defined in:
- System prompt: `LEXam/customized_judge_async.py:24`
- User prompt template: `LEXam/customized_judge_async.py:25`

Injected during async grading:
- `{'role': 'system', 'content': JUDGE_SYSTEM}`
- `{'role': 'user', 'content': JUDGE_PROMPT.format(...)}`
at `LEXam/customized_judge_async.py:327`

## 3. Parsing/Scoring Mechanics

### Judge score extraction
- Regex expects `[[x.x]]` format and clamps invalid to 0:
  - `LEXam/lighteval/community_tasks/lexam_oq_evals.py:192`
  - `LEXam/customized_judge_async.py:184`

### MCQ scoring (non-judge)
- Lighteval MCQ uses regex extraction of `###A###` style answers and computes accuracy:
  - Match regex and sample scoring: `LEXam/lighteval/community_tasks/lexam_mcq_evals.py:65`, `:139`

## 4. Reproduction Commands in README

Lighteval commands reference custom task files:
- Open questions: `LEXam/README.md:56`
- MCQ: `LEXam/README.md:59`

Reasoning-model direct path:
- Generation with `litellm_eval.py`: `LEXam/README.md:83`, `:84`
- Grading with `customized_judge_async.py`: `LEXam/README.md:93`

## 5. Notable Quirks

1. In `customized_judge_async.py`, `Model's Answer` is wrapped as:
   - ````[{model_answer}]````
   at `LEXam/customized_judge_async.py:58`
   while lighteval judge wrapper uses plain ````{answer}```` at
   `LEXam/lighteval/community_tasks/lexam_oq_evals.py:221`.

2. MCQ lighteval path is metric-based accuracy, not LLM-judge based.

3. README typo:
   - Uses `open_quesitons` in example (`LEXam/README.md:83`)
   - Code expects `open_questions` (`LEXam/litellm_eval.py:295`)
