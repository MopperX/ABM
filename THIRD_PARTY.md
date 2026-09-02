# Third-party software, datasets and models

The original benchmark code in this repository is licensed under the **Apache License 2.0**; see the top-level `LICENSE` file. The items documented below are third-party works and remain subject to their own upstream licenses.

Benchmark does not claim authorship of the external benchmark datasets/evaluators used by Core LLM.
They are downloaded during preflight into the persistent benchmark cache outside this repository.

## IFEval

- Project: Google Research Instruction Following Evaluation (IFEval)
- Source: https://github.com/google-research/google-research/tree/master/instruction_following_eval
- License: Apache License 2.0 in the upstream Google Research repository
- Benchmark uses the upstream prompts and strict/loose evaluator.

## TruthfulQA

- Project: TruthfulQA
- Source: https://github.com/sylinrl/TruthfulQA
- License: Apache License 2.0 upstream
- Benchmark uses the recommended Jan-2025 binary-choice setup based on `Best Answer` and `Best Incorrect Answer`.

## MMLU-Pro

- Project: MMLU-Pro
- Dataset: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- Source code: https://github.com/TIGER-AI-Lab/MMLU-Pro
- Dataset license listed by Hugging Face: MIT
- Benchmark uses objective answer accuracy and a deterministic mini subset for quick/standard profiles.

Every run records the external source revisions/hashes used so results remain auditable.


## EvalScope / LiveCodeBench (optional)

- EvalScope 1.11.1 (Apache-2.0) is installed on demand for the optional sandboxed LiveCodeBench integration.
- LiveCodeBench is evaluated through EvalScope's `live_code_bench` adapter. Upstream benchmark data/evaluator artifacts are downloaded at runtime and are not vendored in this repository.


## BEIR / SciFact (RAG retrieval)

- The RAG module downloads the BEIR SciFact dataset at runtime from the upstream BEIR dataset host.
- Dataset files are cached outside the Git repository and are not vendored here.
- The downloaded archive SHA-256 is stored with each run for auditability.
- Benchmark reporting uses nDCG, Recall, Precision and MRR computed locally from the retrieved rankings/qrels.

## Vision datasets

### ScreenSpot

- Project: SeeClick / ScreenSpot
- Dataset used by preflight: `rootsautomation/ScreenSpot`
- License reported by dataset card: Apache-2.0
- Purpose: external GUI-grounding reference
- The benchmark freezes the first resolved Hugging Face commit SHA in its persistent cache.

### MMMU-Pro

- Project: MMMU-Pro
- Dataset: `MMMU/MMMU_Pro`, configuration `standard (10 options)`
- License reported by dataset card: Apache-2.0
- Purpose: external multimodal understanding/reasoning reference
- The benchmark freezes the first resolved Hugging Face commit SHA in its persistent cache.


## Image Generation benchmarks

### GenEval2

- Project: facebookresearch/GenEval2
- Source: https://github.com/facebookresearch/GenEval2
- License: CC BY-NC 4.0 upstream
- The benchmark downloads the official prompt/VQA data during preflight at pinned revision `a6e82d2289e8d418f27f0adee77908b07060eea3`.
- Benchmark uses the official prompts/atom ground truth but a fixed local Ollama Vision judge for universal execution. Results are therefore labelled `GenEval2 local-VQA`, not official Soft-TIFA.

### HPS v2.1

- Project: HPSv2
- Source: https://github.com/tgxs002/HPSv2
- PyPI package: `hpsv2==1.2.0`
- License: Apache-2.0 upstream
- Official benchmark prompts and HPS v2.1 preference scoring are used. Evaluator assets are cached outside Git during preflight.

### Diffusers generation models

Image model weights are not redistributed by this repository. Enabled Hugging Face models are downloaded directly from their upstream repositories during preflight. Each model remains subject to its own upstream license.

## Speech & Audio

### whisper.cpp

- Project: `ggml-org/whisper.cpp`
- Source: https://github.com/ggml-org/whisper.cpp
- Benchmark pin: `v1.8.7`; the resolved Git commit is recorded in the persistent preparation metadata.
- License: MIT upstream.
- Used as the common local Whisper STT inference backend.

### FLEURS (Dutch)

- Dataset: `google/fleurs`, configuration `nl_nl`, split `test`
- Pinned revision: `73c36572c7f01dea15fe27266e26c29f4cda9a83`
- License reported by the dataset card: CC BY 4.0.
- Used as the external Dutch ASR reference and as source material for fixed practical audio fixtures.

### sherpa-onnx

- Project: `k2-fsa/sherpa-onnx`
- Python package pin: `sherpa-onnx==1.13.7`
- License: Apache-2.0 upstream.
- Used for local VITS/Piper TTS and fixed speaker diarization.

### Dutch Piper/VITS voices

The benchmark downloads converted Piper voice models from the official sherpa-onnx release assets. The default voices are `vits-piper-nl_NL-alex-medium` and `vits-piper-nl_NL-pim-medium`, converted from the corresponding `rhasspy/piper-voices` models. Model weights are not redistributed in this Git repository and remain subject to their upstream model/dataset licenses.

### Speaker diarization models

The fixed diarization pipeline downloads the sherpa-onnx Pyannote segmentation conversion and 3D-Speaker embedding model from official sherpa-onnx release assets. These assets are cached outside Git; their upstream model/license terms remain applicable.

## Music Generation

### MusicGen

- Default model: `facebook/musicgen-small`
- Benchmark revision: `f37b0c0576a5fb6891df4a25c80680af72c11e1e`
- Upstream model license: CC-BY-NC-4.0.
- Optional MusicGen models are downloaded directly from their Hugging Face repositories and remain subject to their upstream licenses.
- Benchmark redistributes no MusicGen weights.

### MusicBench

- Dataset: `amaai-lab/MusicBench`
- File used: `MusicBench_test_B.json`
- Benchmark revision: `b141e962aacc19ffd51c15732738040377989203`
- Dataset card license: CC-BY-SA-3.0.
- Benchmark downloads only the test JSON metadata/prompts; it does not download or redistribute the 16.8 GB MusicBench audio archive.
- The local CLAP/BPM/key evaluation in Benchmark is not presented as the official Mustango/MusicBench leaderboard metric.

### LAION CLAP

- Evaluator model: `laion/clap-htsat-fused`
- Benchmark revision: `365dea6ef167def6676140ed93bbc43f84dabb28`
- License: Apache-2.0 upstream.
- Used as a fixed local text/audio similarity evaluator; model weights are cached outside Git during attached preflight.


## SearXNG (Web / Research)

- Project: `searxng/searxng`
- Source: https://github.com/searxng/searxng
- Benchmark revision: `d226b78bc` (resolved to the full Git commit during preflight and stored in the persistent preparation metadata).
- License: AGPL-3.0-or-later upstream.
- SearXNG runs locally on loopback and is used only as the free metasearch backend for the Web / Research suite. No SearXNG source or third-party search result is redistributed in benchmark results beyond the local raw evidence captured by the user's own run.