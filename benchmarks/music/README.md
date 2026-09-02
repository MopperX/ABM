# Music Generation

This suite measures local text-to-music and optional melody-conditioned generation. Quality, control, performance, and listening review remain separate.

The external layer uses pinned MusicBench test-B prompts. Generated ten-second clips are evaluated with a pinned CLAP model on CPU/float32. Tempo and key results are local signal-analysis proxies, not an official MusicBench leaderboard score. All WAV files remain available for blind listening review.

Raw results retain the prompt, deterministic seed, exact model revision, device, duration, generation time, real-time factor, GPU telemetry, CLAP score, music-control proxies, and human-review fields.
