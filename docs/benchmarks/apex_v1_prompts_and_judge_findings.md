# APEX v1 Prompt & Judge Findings

This document summarizes what prompts are used for:
- Question/response generation
- Judge/grading

based on the upstream APEX benchmark codebase (`apex-evals`), not the local runner implementation in this repository.

Scope note: file paths in this document reference the external benchmark repository layout.

## 1. Primary APEX-v1 stack in this repo: `apex-evals-v1-extended`

The top-level repo README points to `apex-evals-v1-extended` as the main APEX-v1 package:
- `apex-evals/README.md:11`

## 2. Question Prompt (Generation) in `apex-evals-v1-extended`

### Prompt template file
- `apex-evals/apex-evals-v1-extended/prompt/response_generation_prompt.txt:1`

This template includes placeholders:
- `{{Domain}}`
- `{{Prompt}}`

### Where it is loaded and filled
- Template loaded from disk:
  - `apex-evals/apex-evals-v1-extended/examples/run_with_hf.py:22`
- Dataset row values are injected:
  - `{{Domain}} <- Domain`
  - `{{Prompt}} <- Prompt`
  - `apex-evals/apex-evals-v1-extended/examples/run_with_hf.py:155`
- Task data source:
  - `data/train.csv`
  - `apex-evals/apex-evals-v1-extended/examples/run_with_hf.py:267`

### Final prompt assembly
- Base prompt is `task.prompt`
- If attachments were parsed, they are appended under:
  - `==== Attached files content: ====`
- Code:
  - `apex-evals/apex-evals-v1-extended/src/generation/executor.py:351`

### Important nuance about "SYSTEM_PROMPT"
- `response_generation_prompt.txt` begins with the text `SYSTEM_PROMPT`, but in the default runner this file is passed as **user content** (not model system role).
- In `run_with_hf.py`, `GenerationTask` is created without `system_prompt`:
  - `apex-evals/apex-evals-v1-extended/examples/run_with_hf.py:110`
- `system_prompt` is optional and defaults to `None`:
  - `apex-evals/apex-evals-v1-extended/src/generation/config.py:42`
- A real system message is only sent if `task.system_prompt` is set:
  - `apex-evals/apex-evals-v1-extended/src/generation/executor.py:193`

## 3. Judge Prompt (Grading) in `apex-evals-v1-extended`

### Default grading prompt template
- Default path:
  - `apex-evals/apex-evals-v1-extended/src/grading/executor.py:18`
- Loaded from:
  - `apex-evals/apex-evals-v1-extended/prompt/grading_prompt.txt:1`
- If no custom template is provided, default is used:
  - `apex-evals/apex-evals-v1-extended/src/grading/executor.py:477`

### How the judge prompt is formed
- The prompt is formatted per criterion with:
  - `{criterion_description}`
  - `{solution}`
  - plus any other keys in the criterion dict
- Code:
  - `apex-evals/apex-evals-v1-extended/src/grading/executor.py:146`

### Where rubric input comes from in the end-to-end script
- Rubric read from dataset column:
  - `Rubric JSON`
  - `apex-evals/apex-evals-v1-extended/examples/run_with_hf.py:154`
- Grading task call:
  - `apex-evals/apex-evals-v1-extended/examples/run_with_hf.py:127`

## 4. Related but separate stack in this repo: `ace/`

If "APEX v1" was intended to include the `ace` harness (same repo, separate pipeline), prompt/judge behavior is different:

### Question prompt in `ace`
- Uses `Specified Prompt` from dataset as the prompt for model calls:
  - `apex-evals/ace/pipeline/init_from_dataset.py:189`
  - `apex-evals/ace/harness/make-grounded-call.py:88`

### Judge prompts in `ace`
- Judging is done via multiple hardcoded autograder prompts (Stage 1/Stage 2/non-grounding/link checks), not a single external template file.
- Examples:
  - Stage 1 response-text criterion prompt:
    - `apex-evals/ace/harness/autograder.py:93`
  - Stage 2 holistic grounding verification prompt:
    - `apex-evals/ace/harness/autograder.py:336`
  - Stage 2 per-product source verification prompt:
    - `apex-evals/ace/harness/autograder.py:412`
  - Non-grounding criterion prompt:
    - `apex-evals/ace/harness/autograder.py:729`

## 5. Bottom line

For the documented APEX-v1 harness (`apex-evals-v1-extended`):
- Generation prompt = `prompt/response_generation_prompt.txt` + dataset `Domain/Prompt` + optional parsed attachments
- Judge prompt = `prompt/grading_prompt.txt` (unless overridden)
- Rubric = dataset-provided `Rubric JSON`
