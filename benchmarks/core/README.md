# Core LLM

Core LLM is offline from the model's perspective: no web tools are exposed.
External benchmark datasets are downloaded/cached during the attached preflight, before the headless run starts.

## Practical tests

- G1 technical diagnosis
- G2 hypothesis ranking
- G3 decision under constraints
- G4 incident handover
- G5.1-G5.10 knowledge boundaries / hallucination resistance
- G6 multi-turn consistency

`quick` runs a smaller G5 subset once. `standard` and `full` run all practical tests three times.

## External reference layer

- IFEval: official Google Research prompts and official strict/loose checker implementation.
- TruthfulQA: Jan-2025 recommended binary-choice format using `Best Answer` vs `Best Incorrect Answer`.
- MMLU-Pro: objective multiple-choice questions across 14 categories.

Profiles:

| Profile | IFEval | TruthfulQA | MMLU-Pro |
|---|---:|---:|---:|
| quick | 10 | 10 | 14 (1/category) |
| standard | 25 | 40 | 42 (3/category) |
| full | all | all | all |

The mini subsets are deterministic. The exact selected keys/question IDs are written into each summary/raw run so the dashboard can show and verify them later.

### Source locking

- IFEval is pinned to a fixed Google Research Git revision.
- MMLU-Pro is pinned to a fixed Hugging Face dataset revision.
- TruthfulQA is cloned only on first cache creation and is **not automatically pulled afterwards**. The exact commit and SHA-256 are recorded in `external_sources.json` for every run. Removing the TruthfulQA cache is an explicit act that can change that source revision.

The Core preflight caches third-party source/data under the persistent benchmark cache outside the Git repository.

## Reasoning modes

Every supported reasoning mode is evaluated as an independent configuration. Examples:

- `nothinking` -> Ollama `think: false`
- `thinking` -> Ollama `think: true`
- `low`, `medium`, `high`, `max` -> Ollama string reasoning level
- `standard` -> no `think` field is sent

Prompts, temperature, seed and context are held constant between reasoning modes. Only the reasoning mode changes.

The final answer is scored. Thinking output is retained in raw data but never counted as the answer.
