# AI Benchmark v4

Release candidate: **4.0.0-rc1**. Zie `RELEASE_NOTES.md`.

Een reproduceerbare, headless benchmark-suite voor lokale AI-modellen op:

- Ubuntu / Debian
- Ubuntu onder WSL
- macOS

De Git-repository bevat **geen resultaten**. Alle runstate, logs, raw output en compacte summaries worden persistent buiten de repository op de rootdisk opgeslagen.

## Belangrijke ontwerpregels

- Repository-updates gebeuren **alleen handmatig** met `git clone` / `git pull`.
- De benchmark voert nooit zelf `git pull` uit.
- Ontbrekende dependencies en Ollama-modellen worden tijdens setup/preflight automatisch geïnstalleerd of gedownload.
- De eigenlijke benchmark draait headless; SSH of de startende terminal mag daarna worden gesloten.
- `thinking`, `nothinking` en reasoning-effort-niveaus zijn afzonderlijke benchmarkconfiguraties.
- Raw data blijft altijd beschikbaar naast de afgeleide summary.
- Webtoegang is per model configureerbaar. Core LLM geeft modellen geen webtools; alleen de expliciete Web / Research-suite geeft tools aan modellen met `web=true`.

## 1. Repository downloaden

Vervang `<GIT-URL>` door de publieke Git-URL.

### Ubuntu / Debian / WSL Ubuntu

```bash
git clone <GIT-URL> ai-benchmark-v4
cd ai-benchmark-v4
chmod +x bootstrap.sh
./bootstrap.sh
```

`bootstrap.sh` installeert de basisdependencies via `apt`, installeert Ollama wanneer nodig, maakt de persistente resultaatmap aan en zet alle scripts uitvoerbaar.

### macOS

```bash
git clone <GIT-URL> ai-benchmark-v4
cd ai-benchmark-v4
chmod +x bootstrap.sh
./bootstrap.sh
```

Als Homebrew ontbreekt, bootstrappt het script Homebrew. Daarna worden onder andere Python, `dialog` en Ollama geïnstalleerd. De headless runner gebruikt `caffeinate` om idle sleep tijdens een actieve run te voorkomen.

## 2. Resultaatlocatie

Standaard:

- Linux / Ubuntu / Debian / WSL: `/var/lib/ai-benchmark-v4/`
- macOS: `/Library/Application Support/ai-benchmark-v4/`

Override is mogelijk:

```bash
BENCH_RESULTS_DIR=/data/ai-results ./benchmark-v4 start core --machine ai-worker-hp
```

Een run ziet er bijvoorbeeld zo uit:

```text
/var/lib/ai-benchmark-v4/runs/ai-worker-hp/20260901T203000Z-ab12/
├── run.json
├── state.json
├── benchmark.log
├── config/
│   ├── machine.models.tsv   # Ollama-modelsnapshot voor deze run
│   ├── image.models.tsv     # Diffusers image-modelsnapshot voor deze run
│   ├── speech.models.tsv    # STT/TTS-modelsnapshot voor deze run
│   └── music.models.tsv     # Music-generation model snapshot voor deze run
├── source/                  # volledige broncode/fixturesnapshot van deze run
├── summary/
│   └── summary.json
└── raw/
    ├── machine.json
    ├── source.json          # Git commit + bron/config-hashes
    └── core/
        └── <model>__<reasoning-mode>/...
```

## 3. Modelconfiguratie per machine

Machinebestanden staan in:

```text
config/machines/
```

Formaat is tab-separated:

```text
enabled  model       benchmarks          modes                 web    notes
true     qwen3:8b    core,coding-agent   nothinking,thinking   false  General model
```

De daadwerkelijke bestanden gebruiken tabs, niet spaties.

Beschikbare reasoningmodi:

- `standard` — geen Ollama `think`-veld
- `nothinking` — `think: false`
- `thinking` — `think: true`
- `low`, `medium`, `high`, `max` — reasoning-effort voor modellen die dit ondersteunen

Dezelfde modelnaam kan dus automatisch meerdere volledig gescheiden benchmarkruns krijgen.

### Online modelselectie aanpassen

Bewerk bijvoorbeeld in Git:

```text
config/machines/ai-worker-hp.models.tsv
```

Commit de wijziging. Op de worker:

```bash
cd ai-benchmark-v4
git pull
```

De volgende run gebruikt de nieuwe lijst. Ontbrekende enabled modellen worden tijdens preflight automatisch met `ollama pull` opgehaald.

Image-generationmodellen hebben bewust een aparte per-machine lijst onder `config/image-models/<machine>.models.tsv`. Speechmodellen staan apart onder `config/speech-models/<machine>.models.tsv` en musicmodellen onder `config/music-models/<machine>.models.tsv`. Alle drie configuraties worden bij het starten in de run gesnapshot; ingeschakelde modelassets worden tijdens attached preflight gecachet.

## 4. Interactief starten

```bash
./benchmark-v4
```

Er verschijnt een terminal-checklist waarin Core, Coding & Agent, RAG, Vision, Image, Speech, Music, Web / Research of `ALL` geselecteerd kunnen worden. Daarna kies je `quick`, `standard` of `full` en de machineconfig.

## 5. Direct starten

```bash
./benchmark-v4 start core --profile standard --machine ai-worker-hp
```

Meerdere benchmarks:

```bash
./benchmark-v4 start core rag vision --profile standard --machine ai-worker-hp
```

Alles:

```bash
./benchmark-v4 start all --profile standard --machine ai-worker-hp
```

Na de preflight wordt de benchmark losgekoppeld van de terminal. De client-pc waarmee je via SSH verbonden bent mag daarna uit.

> Onder WSL moet uiteraard de Windows-host waarop WSL zelf draait ingeschakeld blijven.

## 6. Status

Laatste run:

```bash
./benchmark-v4 status
```

Specifieke run:

```bash
./benchmark-v4 status 20260901T203000Z-ab12
```

De status toont onder andere benchmark, model, reasoningmodus, test, repeat en voortgang.

## 7. Stoppen

Graceful stop:

```bash
./benchmark-v4 stop
```

De runner rondt waar mogelijk de huidige model/backend-call af, bewaart het resultaat en stopt daarna op een checkpoint. Lange prepare/index/generationlussen controleren daarnaast tussentijds op een stopverzoek.

Geforceerd:

```bash
./benchmark-v4 stop --force
```

## 8. Hervatten

```bash
./benchmark-v4 resume
```

Of:

```bash
./benchmark-v4 resume <run-id>
```

Afgeronde `model × reasoningmodus × test × repeat`-combinaties worden niet opnieuw uitgevoerd. Iedere nieuwe run bewaart zowel zijn modelconfiguraties als een volledige snapshot van de benchmarkbroncode en fixtures onder `source/`. Een latere `git pull` kan daardoor een lopende of hervatte run niet halverwege van prompts/tests laten veranderen; `resume` gebruikt dezelfde bron-snapshot. Oude runs van vóór deze snapshotfunctie blijven voor compatibiliteit de oorspronkelijke Git-commitcontrole gebruiken.

## 9. Core LLM — huidige implementatie

Core bestaat uit twee lagen en draait vanuit het perspectief van het model volledig offline. Externe datasets worden tijdens de preflight gedownload/gecached; het model krijgt geen webtool.

### Praktijktests

- G1 technische diagnose
- G2 hypotheses rangschikken
- G3 beslissing onder beperkingen
- G4 incident-handover
- G5.1–G5.10 kennisgrenzen / hallucinatiebestendigheid
- G6 multi-turn consistentie

`standard` voert elke praktische testcase drie keer uit. `quick` gebruikt één repeat en een kleinere G5-subset.

### Externe standaardlaag

| Profiel | IFEval | TruthfulQA binary | MMLU-Pro |
|---|---:|---:|---:|
| `quick` | 10 | 10 | 14 |
| `standard` | 25 | 40 | 42 |
| `full` | volledige set | volledige set | volledige set |

IFEval wordt met de officiële strict/loose checkers beoordeeld. TruthfulQA gebruikt de door het project sinds januari 2025 aanbevolen binaire opzet: één `Best Answer` en één `Best Incorrect Answer`, met deterministisch gewisselde A/B-volgorde. MMLU-Pro rapporteert accuracy totaal en per categorie.

De mini-subsets zijn deterministisch en de exacte geselecteerde IDs worden in de run opgeslagen. IFEval en MMLU-Pro gebruiken vaste bronrevisions; van TruthfulQA wordt de exacte eerste lokaal gecachete Git-commit plus SHA-256 in iedere run vastgelegd. De benchmark doet daarna geen automatische `git pull` op die externe cache.

Voor elke externe vraag wordt de volledige Ollama request/response, thinking-output, timing en beschikbare power-sampling onder `raw/` opgeslagen.

De response van Ollama wordt dus volledig bewaard, inclusief afzonderlijke `message.thinking` wanneer het model die retourneert. Alleen `message.content` wordt inhoudelijk beoordeeld.

Zie ook `benchmarks/core/README.md` en `THIRD_PARTY.md`.

## 10. Coding & Agent — huidige implementatie

Coding & Agent is opgesplitst in twee onafhankelijke delen.

### Code-chat

- C1 Laravel endpoint / Form Request / Pest
- C2 security + transaction repair
- C3 Eloquent N+1 / aggregate / pagination
- C4 race-safe browser live search
- C5 concrete code review
- C6 conversational debugging
- C7 Laravel + Livewire
- C8 Livewire + project Web Awesome wrappers
- C9 project-specific component conventions

`standard` draait C1–C9 drie keer per `model × reasoning-mode`. Alle machine-readable checks en de volledige antwoorden blijven in raw output beschikbaar.

### Repository agent

- A1 kleine bugfix
- A2 feature in bestaande repository
- A3 debugging + test recovery
- A4 grotere repository / contract discovery
- A5 Laravel/Livewire/Web Awesome repository task

De agent krijgt alleen gecontroleerde repositorytools. Hidden tests onder `.benchmark/` worden niet aangeboden via list/read/search-tools. Iedere eindpatch wordt objectief door de hidden tests beoordeeld. Raw output bevat het complete tooltranscript, de finale workspace, testlogs en Git diff. Omdat agenttests modelgewijzigde code uitvoeren, hoort deze module op een dedicated worker zonder productiegeheimen te draaien; zie `SECURITY.md`.

Profielen:

| Profiel | Code-chat | Agent |
|---|---|---|
| `quick` | 6 geselecteerde tests ×1 | A1 + A5 ×1 |
| `standard` | C1–C9 ×3 | A1–A5 ×1 |
| `full` | C1–C9 ×3 | A1–A5 ×3 |

Zie `benchmarks/coding-agent/README.md`.

### Externe coding/agentbenchmarks

De universele praktische Coding & Agent-module is geïmplementeerd. LiveCodeBench is als expliciet optionele, geïsoleerde externe laag geïntegreerd via `--with-livecodebench`; SWE-bench blijft een afzonderlijke optionele zware toekomstige laag en is geen stille dependency van `standard`.

## 11. RAG / Documents

RAG is geïmplementeerd met R1-R7, een vaste `embeddinggemma`-retriever, PDF-fixtures en een externe BEIR SciFact-retrievalcontrole. Retrieval en answerkwaliteit worden apart gerapporteerd; zie `benchmarks/rag/README.md`.

## 12. Vision / Screenshots

Vision is geïmplementeerd met zeven praktische tests (V1–V7) en twee externe referentielagen: ScreenSpot voor GUI-grounding en MMMU-Pro voor multimodale reasoning. De eerste Vision-preflight installeert de Python image/datasetdependencies en bouwt een persistente externe datasetcache buiten de Git-repository. Houd rekening met ongeveer 1–2 GB eerste download/cache voor de externe datasets, afhankelijk van Hugging Face caching/compressie.

De gemeenschappelijke Vision-baseline gebruikt de officiële `qwen3-vl:4b-instruct` én `qwen3-vl:4b-thinking` tags. Qwen3-VL publiceert deze als aparte modelvarianten; de benchmark forceert daarom niet kunstmatig `think:false` op de thinking-gewichten. Zwaardere 8B-varianten en hybride multimodale modellen kunnen per machine online worden ingeschakeld.

Zie `benchmarks/vision/README.md`.

## 13. Image Generation

Image Generation is geïmplementeerd met I1-I7, vaste 512×512 prompts/seeds, per-machine Diffusers-modelconfiguraties en volledige PNG/raw-output. Promptvolging en voorkeur worden apart gehouden van generatiesnelheid.

De externe laag gebruikt officiële GenEval2 prompt/atomdata in een **local-VQA** variant met één vaste Vision-evaluator; dit wordt bewust niet als officiële Soft-TIFA-score gepresenteerd. Daarnaast gebruikt de benchmark HPS v2.1 (`hpsv2` 1.2.0) en de officiële HPS benchmarkprompts. Zie `benchmarks/image/README.md`.

Standaard staat alleen `segmind/tiny-sd` aan als lichte universele baseline. Zwaardere modellen kunnen per machine in Git worden ingeschakeld. De preflight downloadt alle ingeschakelde image-modeldata en evaluatordata vóór de headless runner wordt losgekoppeld.

## 14. Speech & Audio

Speech & Audio is geïmplementeerd met Nederlandse STT, speaker diarization, TTS en een externe FLEURS-NL referentielaag. De module gebruikt per machine een aparte `config/speech-models/*.models.tsv`, bewaart alle WAV/transcripties raw en rapporteert onder andere WER, CER, DER en real-time factor. Zie `benchmarks/speech/README.md`.

## 15. Music Generation

Music Generation is geïmplementeerd met M1-M7, vaste seeds/prompts, per-machine MusicGen-configuraties, volledige raw WAV-output en een externe MusicBench prompt/control-laag. Promptmatching gebruikt een vaste LAION-CLAP evaluator; tempo/key worden als lokale signal-analysis proxies gerapporteerd en niet als officiële MusicBench leaderboardscore.

Standaard staat alleen `facebook/musicgen-small` aan. Melody-conditioned M4 wordt alleen uitgevoerd voor een modelconfig met de capability `melody`; anders wordt die testcase correct als `unsupported` geregistreerd. Zie `benchmarks/music/README.md`.

## 16. Web / Research

Web / Research is de enige suite die modellen internettools aanbiedt. Alleen enabled modellen met `web=true` in `config/machines/<machine>.models.tsv` worden meegenomen. Core en de overige suites blijven vanuit het modelperspectief offline.

De zoeklaag draait lokaal via een vastgepinde SearXNG-installatie op `127.0.0.1`. Tijdens attached preflight worden SearXNG en dependencies automatisch voorbereid; er is **geen betaalde zoek-API en geen API-key** nodig. De suite biedt het model uitsluitend:

- `web_search` via lokale SearXNG;
- `web_fetch` voor publieke HTTP/HTTPS-pagina's.

`web_fetch` blokkeert loopback, private, link-local, reserved en andere niet-publieke IP-adressen om toegang tot lokale services/netwerken te beperken. Search-resultaten, fetches, toolcalls, citations, timings en volledige Ollama-responses blijven onder `raw/web/` beschikbaar.

Tests:

- W1 actuele Ollama release;
- W2 actuele officiële Ollama tool-callingdocumentatie;
- W3 multi-source onderzoek naar de actuele Copilot+ NPU TOPS-eis.

Profielen: `quick` = W1-W2 ×1, `standard` = W1-W3 ×3, `full` = W1-W3 ×5. Omdat het publieke web verandert, worden bron-URLs en opgehaalde content als onderdeel van de raw evidence bewaard. Zie `benchmarks/web/README.md`.

## 17. Optionele externe LiveCodeBench

Coding & Agent kan aanvullend met de geïsoleerde externe LiveCodeBench-laag worden gestart:

```bash
./benchmark-v4 start coding-agent --profile standard --machine ai-worker-hp --with-livecodebench
```

De preflight installeert hiervoor automatisch de benodigde container-runtime en gepinde EvalScope-omgeving. Gegenereerde benchmarkcode wordt door EvalScope in een Docker-sandbox zonder netwerk uitgevoerd. Zonder deze vlag blijven C1-C9 en A1-A5 de normale universele Coding & Agent-suite.

## 18. Handmatig updaten

```bash
cd ai-benchmark-v4
git pull
```

Daarna desgewenst opnieuw:

```bash
./bootstrap.sh
```

De resultaten staan buiten de repository en worden door `git pull` niet geraakt. Nieuwe runs blijven bovendien op hun eigen bron-snapshot draaien, zodat een update geen actieve benchmark inhoudelijk wijzigt.

## 19. Energiegegevens

De runners proberen tijdens modelinference/generation automatisch NVIDIA `power.draw` te samplen wanneer `nvidia-smi` dit ondersteunt. Dit geldt ook voor de Image-, Speech- en Music-modules waar hun lokale backends worden aangeroepen. Raw data bevat gemiddelde/peak GPU-wattage en een benadering van Wh voor de call. Wanneer een platform geen betrouwbare automatische bron aanbiedt wordt energie expliciet als `available: false` opgeslagen; dat is geen benchmarkfout.

## 20. Licentie

De eigen benchmarkcode in deze repository wordt uitgebracht onder de **Apache License 2.0**. Zie `LICENSE` voor de volledige licentietekst.

Externe datasets, modelgewichten, evaluators en andere afhankelijkheden vallen niet automatisch onder Apache-2.0; daarvoor blijven de upstream licenties gelden. Zie `THIRD_PARTY.md`.
