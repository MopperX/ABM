# Image Generation

This suite reports generation performance, resources, prompt adherence, and visual preference separately. It does not produce a universal image-quality score. Generated PNG files remain available for blind human review.

Models are configured with `backend=diffusers` and pinned revisions. The comparable main set uses 512x512 output and deterministic per-prompt seeds. Practical prompt criteria use a fixed local vision evaluator. GenEval2 results are explicitly labelled local-VQA atom accuracy and are not the official Soft-TIFA score. HPS v2.1 scores remain available per prompt and style.

Raw results include exact revision, seed, steps, guidance, device, dtype, wall time, GPU telemetry, evaluator output, and human-review fields.
