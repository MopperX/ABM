# RAG / Documents

De RAG-module meet **retrieval** en **antwoordkwaliteit** afzonderlijk. Het antwoordmodel krijgt
alleen de top-k documenten die de retriever heeft geselecteerd; internet/tools zijn tijdens deze
suite niet beschikbaar.

## Vaste retrieval-laag

Alle machines gebruiken hetzelfde embeddingmodel: `embeddinggemma` via Ollama `/api/embed`.
De praktische documentindex wordt één keer per run opgebouwd en daarna hergebruikt voor alle
antwoordmodellen en reasoningmodi. Zo verandert alleen het antwoordmodel/reasoninggedrag en niet
de retriever tussen vergelijkingen.

Raw output bewaart de volledige documenten, embeddings, rankings, similarities, timings en
power-sampling. De compacte summary bewaart de afgeleide retrievalmetrics en answer pass-rates.

## Praktijktests

| Test | Onderwerp | Kerncontrole |
|---|---|---|
| R1 | Exact feit | juiste passage + bron |
| R2 | Meerdere documenten | drie feiten/bronnen combineren |
| R3 | Ontbrekende informatie | geen cipher verzinnen |
| R4 | Conflicterende documenten | 30/45 dagen + conflict/versionering |
| R5 | Grotere documentset | relevante bron tussen distractors vinden |
| R6 | PDF | feit uit echte PDF-fixture ophalen |
| R7 | Multi-turn | bronnen over twee vervolgvragen correct gebruiken |

Broncitaten hebben bewust een machine-checkbaar formaat: `[SOURCE:bestandsnaam]`.

Profiles:

- `quick`: R1, R3 en R6, elk 1×.
- `standard`: R1-R7, elk 3×.
- `full`: R1-R7, elk 3×; externe retrieval gebruikt de volledige SciFact test-queryset.

Thinking/no-thinking/reasoning-effort wordt net als in Core als aparte configuratie uitgevoerd.

## Externe retrievalreferentie: BEIR SciFact

Tijdens preflight wordt BEIR SciFact naar de persistente benchmark-cache buiten Git gedownload.
De volledige SciFact-corpus wordt met dezelfde vaste embeddinglaag geïndexeerd. `quick` gebruikt
5 vaste queries, `standard` 20 vaste queries en `full` alle beschikbare test-qrels. Selectie is
deterministisch.

De module rapporteert rechtstreeks uit de rankings:

- Recall@5 / Recall@10
- Precision@5 / Precision@10
- MRR@5 / MRR@10
- nDCG@5 / nDCG@10

De SHA-256 van het gedownloade datasetarchief wordt bij de raw resultaten opgeslagen.

## Dependencies

`benchmarks/rag/dependencies.sh` wordt automatisch door preflight uitgevoerd. Het installeert
`pypdf`, downloadt `embeddinggemma` wanneer nodig en cached BEIR SciFact. Er is geen handmatige
Python- of datasetinstallatie nodig na `bootstrap.sh`.
