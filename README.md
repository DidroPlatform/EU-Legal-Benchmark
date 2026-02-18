<p align="center">
  <h1 align="center">⚖️ Legal Benchmark Runner</h1>
  <p align="center">
    <em>A config-driven evaluation pipeline for benchmarking LLMs on European legal reasoning tasks.</em>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-≥3.11-blue?logo=python&logoColor=white" alt="Python 3.11+"></a>
    <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/uv-package_manager-blueviolet?logo=astral" alt="uv"></a>
    <a href="#license"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  </p>
</p>

---

## Overview

**Legal Benchmark Runner** evaluates large language models against curated European legal datasets covering professional reasoning, law exams, multilingual MCQs, and human-rights case law. It supports multiple LLM providers, three judging strategies, and produces structured, reproducible run artifacts.

### Key Features

- 🔌 **Multi-provider** — Route candidate models through NVIDIA NIM, Amazon Bedrock, Mistral, Vercel AI Gateway, or any OpenAI-compatible endpoint via [LiteLLM](https://github.com/BerriAI/litellm)
- 🧑‍⚖️ **Three judging modes** — Rubric-based (LLM-graded criteria), reference-answer (LLM comparison), and MCQ (deterministic exact-match)
- 📊 **Structured outputs** — Every run produces JSONL artifacts, per-dataset summaries, and a full config snapshot for reproducibility
- ⚡ **Parallel & rate-limited** — Configurable worker pools and per-minute rate limits for both generation and judging
- 💾 **Disk caching** — Avoid redundant API calls across re-runs
- ✅ **Validated inputs** — Canonical schema validation fails fast on malformed data

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Artifacts](#output-artifacts)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Dataset Citations](#dataset-citations)
- [License](#license)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        config.yaml                          │
│         (providers · datasets · candidates · judges)        │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Schema Validator │
                    │  (fail-fast)      │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │       Runner Orchestrator    │
              │  (parallel workers + cache)  │
              └──────┬──────────────┬───────┘
                     │              │
          ┌──────────▼──────┐  ┌───▼──────────────┐
          │   Generation    │  │     Judging       │
          │  (LiteLLM /     │  │  (Rubric / Ref /  │
          │   Google GenAI) │  │   MCQ grading)    │
          └─────────────────┘  └──────────────────┘
                     │              │
              ┌──────▼──────────────▼───────┐
              │     Output Writer            │
              │  outputs/<run_id>/           │
              │  examples · responses ·      │
              │  judgments · summary          │
              └─────────────────────────────┘
```

---

## Datasets

The benchmark input is built from five curated upstream sources, each mapped to a specific judging mode:

| Dataset | Source | Task Type | Judging Mode |
|---------|--------|-----------|--------------|
| **PRBench** | [ScaleAI/PRBench](https://huggingface.co/datasets/ScaleAI/PRBench) | Rubric QA | LLM rubric |
| **APEX-v1** | [mercor/APEX-v1-extended](https://huggingface.co/datasets/mercor/APEX-v1-extended) | Rubric QA | LLM rubric |
| **LEXam** | [LEXam-Benchmark/LEXam](https://huggingface.co/datasets/LEXam-Benchmark/LEXam) | Reference QA / MCQ | LLM reference / Exact-match |
| **INCLUDE-Base** | [CohereLabs/include-base-44](https://huggingface.co/datasets/CohereLabs/include-base-44) | MCQ | Exact-match |
| **LAR-ECHR** | [AUEB-NLP/lar-echr](https://huggingface.co/datasets/AUEB-NLP/lar-echr) | MCQ | Exact-match |

All datasets are converted into a canonical `legal_eval_v1` JSONL schema (see [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md)) and merged into a single evaluation file.

---

## Getting Started

### Prerequisites

- **Python ≥ 3.11**
- [**uv**](https://docs.astral.sh/uv/) — fast Python package manager
- API keys for at least one LLM provider (see [Configuration → Providers](#providers))
- If any configured model uses the `bedrock/...` prefix, install `boto3` (included by default via `uv sync`)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/legal-benchmark-runner.git
cd legal-benchmark-runner

# Install dependencies
uv sync

# Copy template files
cp .env.example .env
cp config.example.yaml config.yaml
```

### Set up API keys

Edit `.env` and add the keys for the providers you plan to use:

```bash
# Required for your chosen providers (add only what you need)
NVIDIA_API_KEY=your_nvidia_api_key
MISTRAL_API_KEY=your_mistral_api_key
GEMINI_API_KEY=your_gemini_api_key
AWS_BEARER_TOKEN_BEDROCK=your_aws_bedrock_api_key
AI_GATEWAY_API_KEY=your_vercel_ai_gateway_api_key
```

### Build the eval dataset

Merge curated source datasets into the canonical evaluation file:

```bash
uv run python build_for_eval.py
```

This produces `data/for_eval/merged_legal_eval_v1.jsonl`.

### Verify your setup

```bash
uv run python run.py --config config.yaml --check-setup
```

If everything is configured correctly, you'll see: `Setup check passed.`  
If Bedrock models are configured without `boto3`, setup check fails fast with an install hint.

---

## Configuration

All runtime behavior is controlled through `config.yaml` (or any YAML file passed via `--config`). The [`config.example.yaml`](config.example.yaml) file is a fully annotated template.

### Providers

Define credential profiles and routing settings. Each candidate or judge model references a provider by name.

```yaml
providers:
  nim:
    api_key_env: NVIDIA_API_KEY
    base_url: https://integrate.api.nvidia.com/v1
    timeout_s: 180

  bedrock:
    api_key_env: AWS_BEARER_TOKEN_BEDROCK
    timeout_s: 180

  google_genai:
    api_key_env: GEMINI_API_KEY
    timeout_s: 120
```

### Candidates & Judges

```yaml
candidates:
  - name: bedrock_claude_sonnet_4_5
    provider: bedrock
    model: bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0
    temperature: 0.2
    max_tokens: 4096

judges:
  - name: judge_gemini_flash_lite
    provider: google_genai
    model: gemini-flash-lite-latest
    temperature: 0.0
    max_tokens: 700
```

### Runtime controls

| Parameter | Description | Default |
|-----------|-------------|---------|
| `response_parallel_workers` | Parallel candidate generation workers | `8` |
| `response_rate_limit_rpm` | Shared RPM throttle for generation (1–50) | `50` |
| `provider_response_rate_limit_rpm` | Optional per-provider generation RPM overrides (1–50) | `{}` |
| `judge_parallel_workers` | Parallel judge workers per response | `4` |
| `judge_rate_limit_rpm` | RPM throttle for judge calls (0 = off) | `12` |

Example provider-specific throttle:

```yaml
run:
  response_rate_limit_rpm: 50
  provider_response_rate_limit_rpm:
    nim: 20
```

---

## Usage

```bash
# Full benchmark run
uv run python run.py --config config.yaml

# Smoke test with 5 examples
uv run python run.py --config config.yaml --limit 5

# Validate setup without running
uv run python run.py --config config.yaml --check-setup

# Disable progress output
uv run python run.py --config config.yaml --progress off
```

### CLI Reference

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config file (default: `config.example.yaml`) |
| `--limit N` | Cap total examples across all datasets |
| `--progress {log,off}` | Progress output mode |
| `--check-setup` | Validate environment and exit |

---

## Output Artifacts

Each run writes files to `outputs/<run_id>/`:

| File | Description |
|------|-------------|
| `examples.jsonl` | Normalized examples selected for the run |
| `responses.jsonl` | Candidate model outputs with request metadata |
| `judgments.jsonl` | Grading outputs (score, pass/fail, criteria, rationale) |
| `scored_responses.jsonl` | Merged response + judgment rows |
| `trace.jsonl` | Per-call trace data for debugging |
| `summary.json` | Aggregate metrics (overall and per-dataset) |
| `run_config.json` | Resolved config snapshot for reproducibility |

---

## Testing

The project includes a comprehensive test suite covering schema validation, MCQ grading, rubric aggregation, prompt policies, runner progress, and rate limiting.

```bash
# Run the full test suite
uv run pytest

# Run a specific test file
uv run pytest tests/test_mcq_grading.py -v

# Run with output
uv run pytest -s
```

---

## Project Structure

```
legal-benchmark-runner/
├── run.py                    # CLI entry point — runs the evaluation pipeline
├── build_for_eval.py         # Merges curated datasets into canonical eval file
├── config.example.yaml       # Annotated example configuration
├── .env.example              # Template for API keys
├── pyproject.toml            # Project metadata & dependencies
│
├── src/                      # Core library
│   ├── config.py             # Config parsing & validation
│   ├── types.py              # Shared data types
│   ├── cache.py              # Disk-based response cache
│   ├── retry.py              # Retry logic with exponential backoff
│   ├── setup_checks.py       # Environment validation
│   ├── data/                 # Data loading, schema, attachments
│   │   ├── schema.py         # Canonical JSONL schema validator
│   │   ├── loader.py         # Dataset loader
│   │   ├── build_for_eval.py # Dataset merge/build logic
│   │   ├── policies.py       # Dataset-specific prompting policies
│   │   └── attachments.py    # PDF/file attachment extraction
│   ├── providers/            # LLM provider adapters
│   │   ├── base.py           # Abstract provider interface
│   │   ├── litellm.py        # LiteLLM adapter (OpenAI-compatible)
│   │   └── google_genai.py   # Google GenAI native adapter
│   ├── judge/                # Judging & grading
│   │   ├── judge.py          # Rubric & reference judging
│   │   ├── mcq.py            # Deterministic MCQ grading
│   │   └── parsing.py        # Judge output parsing
│   ├── prompting/            # Prompt construction
│   │   └── templates.py      # Per-policy prompt templates
│   └── runner/               # Execution engine
│       ├── orchestrator.py   # Two-phase orchestrator
│       ├── generation.py     # Candidate generation phase
│       ├── judging.py        # Judging phase
│       ├── output.py         # Artifact writing & summary
│       ├── rate_limiter.py   # Per-minute rate limiter
│       └── helpers.py        # Utility functions
│
├── data/
│   ├── curated/              # Source datasets (JSONL)
│   └── for_eval/             # Canonical merged eval file
│
├── docs/                     # Documentation
│   ├── DATA_SCHEMA.md        # Canonical JSONL schema spec
│   └── POLICIES.md           # Dataset-specific policy docs
│
└── tests/                    # Test suite (pytest)
```

---

## Dataset Citations

<details>
<summary><strong>Click to expand BibTeX entries</strong></summary>

```bibtex
@misc{scaleai2025prbench,
  title        = {PRBench: Large-Scale Expert Rubrics for Evaluating
                  High-Stakes Professional Reasoning},
  author       = {{Scale AI}},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/datasets/ScaleAI/PRBench}},
  note         = {Hugging Face dataset card}
}

@misc{mercor2025apexv1extended,
  title        = {APEX-v1-extended},
  author       = {{Mercor}},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/datasets/mercor/APEX-v1-extended}},
  note         = {Hugging Face dataset card}
}

@article{fan2025lexam,
  title   = {LEXam: Benchmarking Legal Reasoning on 340 Law Exams},
  author  = {Fan, Yu and Ni, Jingwei and Merane, Jakob and Tian, Yang
             and Hermstr{\"u}wer, Yoan and Huang, Yinya and Akhtar,
             Mubashara and Salimbeni, Etienne and Geering, Florian
             and Dreyer, Oliver and Brunner, Daniel and Leippold, Markus
             and Sachan, Mrinmaya and Stremitzer, Alexander and Engel,
             Christoph and Ash, Elliott and Niklaus, Joel},
  journal = {arXiv preprint arXiv:2505.12864},
  year    = {2025}
}

@article{romanou2024include,
  title   = {INCLUDE: Evaluating Multilingual Language Understanding
             with Regional Knowledge},
  author  = {Romanou, Angelika and Foroutan, Negar and Sotnikova, Anna
             and Chen, Zeming and Nelaturu, Sree Harsha and Singh, Shivalika
             and Maheshwary, Rishabh and Altomare, Micol and Haggag,
             Mohamed A and Amayuelas, Alfonso and others},
  journal = {arXiv preprint arXiv:2411.19799},
  year    = {2024}
}

@inproceedings{chlapanis-etal-2024-lar,
  title     = {LAR-ECHR: A New Legal Argument Reasoning Task and Dataset
               for Cases of the European Court of Human Rights},
  author    = {Chlapanis, Odysseas S. and Galanis, Dimitrios
               and Androutsopoulos, Ion},
  booktitle = {Proceedings of the Natural Legal Language Processing
               Workshop 2024},
  year      = {2024},
  address   = {Miami, FL, USA},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2024.nllp-1.22/},
  doi       = {10.18653/v1/2024.nllp-1.22},
  pages     = {267--279}
}
```

</details>

> **Note:** `PRBench` and `APEX-v1-extended` are cited as `@misc` entries because their Hugging Face dataset cards do not publish a BibTeX block.

---

## Troubleshooting

| Symptom | Solution |
|---------|----------|
| Provider env var missing | Set the variable named in `providers.<name>.api_key_env` in your `.env` file |
| Vertex project/location missing | Set `providers.vertex.project` / `location` in config, or export `VERTEXAI_PROJECT` / `VERTEXAI_LOCATION` |
| No examples selected | Verify `data.datasets[*].enabled` is `true` and check `split_field`, `split_value`, and `limit` settings |
| Module import errors | Run from the repo root so `src` is importable (e.g., `uv run python run.py`) |

---

## License

The source code in this repository is licensed under the [MIT License](LICENSE).

> **Data licensing disclaimer:** The curated evaluation data included in `data/curated/` and `data/for_eval/` is derived from the upstream datasets listed in [Dataset Citations](#dataset-citations). That data is **not** covered by this repository's MIT license. Each upstream dataset remains subject to its own license and terms of use; review the corresponding Hugging Face dataset cards before any redistribution or commercial use.
