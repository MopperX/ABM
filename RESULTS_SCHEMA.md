# Results schema

All output fields and statuses are English. Schema version 1 remains backward compatible; new fields are additive.

## Run metadata

- `raw/source.json` freezes benchmark and configuration hashes.
- `raw/machine.json` records the host and runtime.
- `raw/models/ollama.json` records digest, size, quantization, capabilities, model context, template, and parameters.

## Per-call evidence

- `request` and `response` retain exact inputs and outputs.
- `metrics` contains token counts, durations, throughput, and runtime placement.
- `power` contains GPU telemetry and `estimated_gpu_energy_wh`.
- `_benchmark_runtime` contains `/api/ps` identity, allocated context, and VRAM residency after warm-up.

## Quality

- `checks_passed` and `checks_total` are atomic contract results.
- `contract_score` is their ratio.
- `full_contract_pass` requires every required check.
- `critical_failure` denotes a failed critical check.
- `confidence_interval_95` contains Wilson bounds where available.

`pass: null` means no pass threshold is defined. `status: unsupported` is not a failure and remains visible in coverage reporting.

Distribution objects contain `count`, `mean`, `median`, `minimum`, `maximum`, `standard_deviation`, and `coefficient_of_variation`.
