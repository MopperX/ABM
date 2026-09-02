# Security notes

AI Benchmark v4 is intended for dedicated local AI workers.

## Coding & Agent executes generated/modified code

The Coding & Agent benchmark allows a local model to edit disposable fixture repositories and run their tests. The final hidden-test pass/fail step also executes the model-modified fixture code.

The harness limits repository tool paths to the disposable workspace, does not expose `.benchmark/` through list/read/search tools, and strips the worker's normal environment variables from fixture test processes. This reduces accidental credential leakage, but it is **not a hardened operating-system sandbox**.

Recommended practice:

- run the suite on a dedicated benchmark/AI worker;
- do not keep production secrets, SSH private keys, cloud credentials or sensitive data in the benchmark process environment;
- do not run Coding & Agent as `root`;
- treat model-generated code like any other untrusted code;
- use an isolated VM/container/host if stronger isolation is required.

External code-execution benchmarks such as LiveCodeBench are only enabled explicitly and use documented execution isolation; they do not silently execute arbitrary generated programs as part of the normal universal profile.

## Optional LiveCodeBench isolation

When `--with-livecodebench` is selected, LiveCodeBench is evaluated through EvalScope 1.11.1 with its Docker sandbox enabled. The sandbox is configured with networking disabled and 1 CPU / 1 GB RAM limits for generated-code execution. Dataset/evaluator downloads happen outside the generated-code sandbox during dependency/evaluation setup.

The external layer is opt-in because it installs/starts a container runtime and is substantially heavier than the practical C1-C9/A1-A5 suite.


## Image model trust

The Image Generation module loads third-party Hugging Face model repositories with PyTorch/Diffusers. Only enable image models from repositories you trust. Some older Diffusers repositories can contain legacy PyTorch `.bin`/pickle weights rather than safetensors. The public Git config is intentionally human-reviewable and benchmark code never auto-enables newly discovered models.

All enabled image models are downloaded during attached preflight; the headless generation phase uses the cached model snapshot. Gated repositories require the user to accept the upstream license manually before enabling them.

## Speech/audio data

The Speech & Audio benchmark processes audio locally after preflight. Practical/generated WAV files, FLEURS cache files, transcripts and TTS outputs are stored in the persistent benchmark data directory outside Git and are not uploaded by the benchmark runner. Preflight accesses the network only to install dependencies and download the configured public models/datasets.

## Music model trust and audio privacy

The Music Generation module loads third-party Hugging Face model repositories through Transformers/PyTorch. Only enable model repositories you trust. The public per-machine music configuration is deliberately human-reviewable; newly discovered models are never enabled automatically.

Music prompts and generated WAV files remain local in the persistent benchmark result directory. The benchmark does not upload generated music. Network access during music preflight is limited to installing dependencies and downloading the explicitly configured public models/evaluator/dataset metadata; the later headless generation phase uses the cached assets.


## Web / Research network isolation

Only the explicit Web / Research suite exposes network tools to a model, and only for model rows with `web=true`. Other benchmark suites do not expose these tools. Search goes through a loopback-only SearXNG instance. `web_fetch` validates HTTP/HTTPS destinations before connecting and again after redirects; loopback, private, link-local, multicast, reserved and unspecified IP destinations are blocked to reduce SSRF access to the worker or LAN.

This is a defense-in-depth restriction, not a perfect network sandbox. Run web-enabled agent/model evaluation on a dedicated worker and do not expose sensitive unauthenticated services to that host.
