# RAG / Documents

This suite measures retrieval and grounded answer quality separately. All configurations use the same `embeddinggemma` retrieval layer and receive only the selected top-k documents.

Practical tests cover exact facts, multi-document synthesis, missing information, conflicting revisions, distractors, PDF extraction, and multi-turn grounding. Citations use `[SOURCE:filename]` for deterministic validation.

BEIR SciFact provides the external retrieval reference with Recall, Precision, MRR, and nDCG at 5 and 10. Quick uses 5 deterministic queries, standard 20, and full all available test qrels. Dataset hashes, selected IDs, embeddings, rankings, timings, and power evidence are retained.

Quick runs R1/R3/R6 once, standard runs R1-R7 three times, and full runs R1-R7 five times. Repeats measure execution and quality stability, not additional independent questions.
