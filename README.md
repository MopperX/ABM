# AI BenchMark

AI Benchmark evaluates deployable **machine + model configurations**. It measures the quality a model delivers on a specific machine alongside throughput, latency, energy, memory use, and stability. It is neither a hardware-only benchmark nor a hardware-independent model leaderboard.

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

### Prerequisites

- Ubuntu/Debian, Ubuntu on WSL2, or macOS.
- Bootstrap installs or updates Ollama to the latest stable release automatically, verifying its SHA-256 digest from the corresponding GitHub release metadata. Set `OLLAMA_INSTALL_ROOT` to install it outside `/usr/local`.
- On macOS, bootstrap installs [Homebrew](https://brew.sh/) when needed. On Ubuntu/Debian and WSL2, bootstrap uses `sudo` to install OS packages with `apt`.

The bootstrap script installs benchmark dependencies and creates the results directory. Use `./bootstrap.sh --update` to upgrade installed apt or Homebrew packages, Ollama, and compatible Python packages.

Use `./scripts/update-python-dependencies.sh` to update Python packages to their newest versions within the declared compatibility ranges. It refreshes the Core lock for the active Python version and updates existing isolated Web and LiveCodeBench environments. Fixed evaluator versions remain pinned to preserve their scoring contract.

Results are stored in `${XDG_STATE_HOME:-$HOME/.local/state}/ai-benchmark` on Ubuntu/Debian and WSL2, and in `$HOME/Library/Application Support/ai-benchmark` on macOS. Set `BENCH_RESULTS_DIR` to use a different location.

Preflight requires at least 20 GiB free on the results filesystem before it downloads dependencies, datasets, and models. Set `BENCH_MIN_FREE_GB` to a different non-negative whole-number threshold when the selected model configuration requires more capacity or for a deliberately small test run.

Preflight warns, without blocking the run, when CPU load, available Linux memory, active Docker containers, or NVIDIA compute processes may affect a raw performance measurement. The observed readiness state is retained in each run's machine metadata.

### Platform support

Supported environments are Ubuntu/Debian, Ubuntu on WSL2, and macOS. Fedora, Arch, Alpine, and other Linux distributions are intentionally unsupported because bootstrap and suite dependency hooks use Debian packages and `apt`.

| Environment | Required host setup | LiveCodeBench |
|---|---|---|
| Ubuntu/Debian | `sudo`, `apt`, Bash, and internet access for the latest stable CPython, dependencies, datasets, and models | Docker is installed and started during its dependency preflight. |
| Ubuntu on WSL2 | WSL2 with an Ubuntu distribution, `sudo`, `apt`, Bash, and internet access | Docker must be usable from the WSL distribution. Docker Desktop WSL integration is supported. |
| macOS | Administrator access for system installation, Bash, and internet access; Homebrew is installed automatically when missing | Docker CLI and Colima are installed during dependency preflight; Colima is started when Docker is unavailable. |

Coding & Agent runs include LiveCodeBench by default. It executes generated code in a Docker sandbox with network access disabled; use `--without-livecodebench` only to skip that external evaluator.

Storage, memory, and accelerator requirements depend on the enabled rows in the selected machine configuration and the chosen profile. Reserve capacity for the selected model files, downloaded datasets, Python environments, benchmark cache, and retained raw results. The `full` profile performs more repeats and complete external datasets, so it requires materially more runtime and storage than `quick` or `standard`.

Bootstrap creates the project environment with the latest stable CPython through `uv`; the LiveCodeBench environment uses the same interpreter. CI validates the repository against Python 3.14 as the current compatibility baseline. Every run records the installed Python, Ollama, Docker, operating-system, and available hardware metadata, so dependency and host updates remain visible in the result evidence.

```bash
./bootstrap.sh
./benchmark start all --profile standard --machine ai-worker-hp
./benchmark status
```

Results are written outside Git by default. Every run freezes its source, machine configuration, dataset manifests, exact Ollama model identity, and Python environment fingerprint. A resume is rejected if a model tag resolves to a different digest or shared Python dependencies have changed.

Preflight refreshes every selected Ollama model tag before a new run or resume. A tag that has moved to a new model digest starts a new run with that updated identity; resume remains intentionally blocked because it must use the original digest.

Enabled Image and Music model rows use the Hugging Face `main` revision and are refreshed during preflight; each run records the resolved revision. Fixed benchmark datasets and evaluator assets remain pinned so their scoring contract does not move with upstream releases.

Quality, performance, efficiency, and stability remain separate dimensions. Reports do not combine unrelated metrics into a universal score. See [METHODOLOGY.md](METHODOLOGY.md), [RESULTS_SCHEMA.md](RESULTS_SCHEMA.md), and [SECURITY.md](SECURITY.md).

## Language policy

Code, documentation, CLI messages, benchmark prompts, metadata, and generated reports are English. The Speech suite defaults to Dutch prompts, source audio, transcripts, and answers, and uses English only when its language setting requests it; its technical field names and statuses remain English.
