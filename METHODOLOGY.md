# Methodology

## Objective

The unit under test is a deployable machine + model configuration. Quality describes the result produced by that configuration; performance describes how the machine executes it.

## Controlled execution

Comparable runs use identical fixtures, prompts, model digest, quantization, context length, reasoning mode, temperature, seed, and runtime settings. Temperature 0 and seed 42 provide deterministic workloads.

1. Preflight validates dependencies and freezes source and dataset metadata.
2. `/api/tags` and `/api/show` record exact Ollama model identity.
3. An unscored warm-up loads each model/mode/context configuration.
4. `/api/ps` records allocated context and VRAM residency.
5. Measured tasks use one, three, or five repeats for quick, standard, or full.
6. Raw evidence is retained and summaries are derived from it.

Repeats measure performance stability. A repeated question counts once as an independent quality item. Differing outcomes are quality instability.

## Metrics and statistics

Quality metrics remain suite-specific. Objective evaluators are primary. Practical regular-expression checks are labelled contract checks and report partial completion, full pass, and critical failures.

Performance reports per-call and whole-task throughput, wall time, and load time. Median is primary, with mean, minimum, maximum, standard deviation, and coefficient of variation.

Binary metrics use 95% Wilson intervals. Continuous metrics use deterministic bootstrap intervals where implemented. Comparisons should be paired over identical item IDs. Quick is indicative, standard is the primary comparison, and full validates finalists.

GPU telemetry includes estimated energy, power, temperature, utilization, clocks, and memory where available. GPU energy must not be described as total system energy. External wall-power measurement remains optional.

## Comparison rules

Hardware comparisons require the same model digest, quantization, context, reasoning mode, benchmark version, and workload. Different models may be compared as complete deployable configurations, but not presented as hardware-only effects.

No universal total score is produced. Scorecards keep quality, performance, efficiency, and stability separate and may identify Pareto-optimal configurations.
