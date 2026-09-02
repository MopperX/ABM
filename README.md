# AI Benchmark v4

AI Benchmark v4 evaluates deployable **machine + model configurations**. It measures the quality a model delivers on a specific machine alongside throughput, latency, energy, memory use, and stability. It is neither a hardware-only benchmark nor a hardware-independent model leaderboard.

## Configuration identity

Every result belongs to the complete combination of machine, accelerator, model digest, quantization, reasoning mode, context length, Ollama/runtime version, and benchmark source version. Direct hardware comparisons are valid only when the model digest, quantization, context, reasoning mode, and workload match.

## Suites

- Core LLM: practical contracts, IFEval, TruthfulQA, and MMLU-Pro.
- Coding & Agent: code-chat checks, repository tasks, and optional LiveCodeBench.
- RAG / Documents: retrieval and grounded answer quality.
- Vision / Screenshots: practical vision, ScreenSpot, and MMMU-Pro.
- Image Generation: speed, resources, prompt adherence, GenEval2 local VQA, and HPS v2.1.
- Speech & Audio: Dutch STT, diarization, and TTS.
- Music Generation: generation performance, CLAP alignment, and music-control proxies.
- Web / Research: local SearXNG-backed research with citation validation.

## Profiles

| Profile | Purpose | Measured practical repeats |
|---|---|---:|
| quick | Compatibility check and indicative result | 1 |
| standard | Primary practical comparison | 3 |
| full | Finalist validation and complete external sets | 5 |

Each LLM configuration receives an unscored warm-up before measured work. Repeats primarily measure performance and stability; they are not additional independent quality questions. Quick quality results are explicitly indicative.

## Running

```bash
./bootstrap.sh
./benchmark-v4 start all --profile standard --machine ai-worker-hp
./benchmark-v4 status
```

Results are written outside Git by default. Every run freezes its source, machine configuration, dataset manifests, and exact Ollama model identity. A resume is rejected if a model tag resolves to a different digest.

Quality, performance, efficiency, and stability remain separate dimensions. Reports do not combine unrelated metrics into a universal score. See [METHODOLOGY.md](METHODOLOGY.md), [RESULTS_SCHEMA.md](RESULTS_SCHEMA.md), and [SECURITY.md](SECURITY.md).

## Language policy

Code, documentation, CLI messages, benchmark prompts, metadata, and generated reports are English. The Speech suite may use Dutch prompts, source audio, transcripts, and answers; its technical field names and statuses remain English.
