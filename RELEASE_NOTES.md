# AI Benchmark v4 — 4.0.0-rc3

Third release candidate for public Git review and real-worker validation.

## Included suites

- Core LLM: practical G1-G6 + IFEval + TruthfulQA + MMLU-Pro
- Coding & Agent: C1-C9 + A1-A5; optional isolated LiveCodeBench
- RAG / Documents: R1-R7 + BEIR SciFact
- Vision / Screenshots: V1-V7 + ScreenSpot + MMMU-Pro
- Image Generation: I1-I7 + GenEval2 local-VQA + HPS v2.1
- Speech & Audio: Dutch STT/TTS/diarization + FLEURS NL
- Music Generation: M1-M7 + MusicBench/CLAP controls
- Web / Research: W1-W3 via local SearXNG; enabled only for model rows with `web=true`

## Runtime behavior

- Ubuntu/Debian, Ubuntu on WSL, and macOS
- automatic attached preflight for dependencies and model assets
- headless execution after preflight
- interactive benchmark selection or CLI `start ...` / `start all`
- `status`, graceful/forced `stop`, and checkpointed `resume`
- separate thinking/no-thinking/reasoning-effort configurations
- persistent results outside Git, with raw evidence and compact summary
- per-run snapshots of model configs **and benchmark source/fixtures**, so a later `git pull` cannot change a running/resumed run

## RC scope

The harness has been syntax-, lifecycle-, fixture-, and mock-inference tested. Several large external model/dataset downloads cannot be fully exercised in the artifact build environment; the first real-worker runs are therefore the final RC validation step before a stable v4.0 release.

The repository's own benchmark code is now licensed under **Apache License 2.0** via the top-level `LICENSE` file. Third-party datasets, models, evaluators, and dependencies retain their upstream licenses as documented in `THIRD_PARTY.md`.


## Changes since rc2

- replaced four separate per-machine model configuration locations with **one unified `config/machines/<machine>.models.tsv`**;
- added `backend` and `suites` columns so Ollama, Diffusers, Whisper, sherpa-onnx TTS and MusicGen rows can coexist cleanly;
- added optional shared columns for modality-specific settings such as revision, capabilities, steps, guidance, language and speaker;
- every run now snapshots only that one complete model configuration;
- updated preflight and all modality runners to filter the unified file by backend + suite;
- removed the old `config/image-models`, `config/speech-models` and `config/music-models` directories.

## Changes since rc1

- selected Apache License 2.0 for the repository's own benchmark code;
- added the canonical top-level `LICENSE`;
- clarified that third-party assets remain governed by their upstream licenses.
