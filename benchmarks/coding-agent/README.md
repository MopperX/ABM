# Coding & Agent

This module separates **code-chat ability** from **repository-agent ability**.
Every supported reasoning mode is a separate configuration; the final answer/patch is evaluated, while thinking output is retained only as raw evidence.

## Code-chat practical tests

- C1 Laravel endpoint + Form Request + Pest contract
- C2 secure transactional controller repair
- C3 Eloquent N+1 / aggregate / pagination
- C4 race-safe browser live search
- C5 code review of untrusted CSV import
- C6 conversational debugging of stale search results
- C7 Laravel + Livewire state/action implementation
- C8 Livewire + project Web Awesome dropdown
- C9 project-specific wrapper/component conventions

Scoring uses fixed machine-readable checks. Every prompt, answer, Ollama response, timing, reasoning output and power sample remains under `raw/`.

## Repository agent tasks

- A1 find and fix a median bug
- A2 add a backwards-compatible status filter
- A3 diagnose a duration parser bug and recover using tests
- A4 navigate a larger repository and obey an authoritative contract
- A5 Laravel/Livewire/Web Awesome project task that requires reading project docs before editing

The model uses a small controlled repository tool protocol: list/search/read/write/replace/run-tests/git-diff/finish. `.benchmark/` hidden tests are inaccessible to the model. Final pass/fail always comes from those hidden tests.

For each agent task the raw output includes the complete tool transcript, final workspace, final test logs and Git diff. Summary fields include resolve rate, tool-call count, test-run count, changed files and diff size.

## Profiles

| Profile | Code-chat | Agent |
|---|---|---|
| quick | C1,C2,C4,C6,C8,C9 ×1 | A1,A5 ×1 |
| standard | C1–C9 ×3 | A1–A5 ×1 |
| full | C1–C9 ×5 | A1–A5 ×5 |


## Reasoning modes

Reasoning modes come from the per-machine model configuration. Only modes actually supported by a model should be listed. For example, the current Qwen3-Coder instruct models are configured as `standard` rather than inventing a thinking/no-thinking pair. General Qwen3 models that support both modes remain separate `thinking` and `nothinking` configurations.

## External benchmark layer

LiveCodeBench and official SWE-bench are intentionally not labeled as completed in this milestone. Official SWE-bench requires a Docker-based reproducible harness and is very storage-heavy; it will be added as an explicit optional external benchmark rather than silently turning the universal standard run into a ~100+ GB Docker workload.

## Default external LiveCodeBench layer

LiveCodeBench runs by default whenever `coding-agent` is selected. Use `--without-livecodebench` only when the external evaluator must be skipped.

```bash
./benchmark start coding-agent --profile standard --machine ai-worker-hp
```

Profiles use 3 (quick), 10 (standard), or the full EvalScope LiveCodeBench dataset (full). Evaluation runs through pinned EvalScope 1.11.1 with a Docker sandbox whose generated-code network access is disabled and whose sandboxes EvalScope removes on completion. EvalScope raw predictions, reviews, reports, logs and progress files are retained under the normal run `raw/` tree.

Reasoning is kept separate: `nothinking` maps to OpenAI-compatible `reasoning_effort=none`, `thinking` to `medium`, and GPT-OSS `low`/`medium`/`high` remain unchanged. This mapping is stored in the raw command metadata.
