# Web / Research

Web / Research is the only benchmark suite that gives selected Ollama models internet tools.

- Core, Coding & Agent, RAG and the other suites remain offline from the model's point of view.
- A model runs Web / Research only when the global catalog has `web=true` and the local machine scan allows it.
- Search is provided by a pinned local SearXNG instance on loopback; there is no paid search API or API key.
- `web_fetch` accepts only public HTTP/HTTPS destinations. Loopback, private, link-local, reserved and local-network addresses are rejected to reduce SSRF risk.
- Every search result, fetched page excerpt, tool call, response, citation and timing is retained under `raw/web/`.

## Tests

- W1: latest stable Ollama release from current web sources.
- W2: current official Ollama tool-calling documentation.
- W3: current Copilot+ NPU TOPS requirement, using Microsoft plus another source when available.

Profiles:

| Profile | Tests | Repeats |
|---|---|---:|
| quick | W1-W2 | 1 |
| standard | W1-W3 | 3 |
| full | W1-W3 | 5 |

The Web suite measures tool use, citation validity, source retrieval and answer checks. Because the public web changes over time, raw search/fetch evidence and timestamps are essential parts of the result.
