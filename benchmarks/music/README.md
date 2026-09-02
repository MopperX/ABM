# Music Generation

Benchmark 7 meet lokale **text-to-music generation** en, waar een model dat ondersteunt, **melody-conditioned generation**. De module draait volledig lokaal nadat de attached preflight dependencies, modelweights, evaluator en vaste MusicBench-prompts heeft gecachet.

## Doel

De benchmark houdt vier dingen uit elkaar:

1. **Prompt adherence** — vaste LAION-CLAP cosine similarity tussen prompt en gegenereerde audio.
2. **Muzikale controle** — tempo/key-indicatoren waar de prompt of externe dataset daarvoor ground truth bevat.
3. **Performance** — model-loadtijd, generation wall time, real-time factor/audio-seconden per generatie-seconde, RAM/VRAM en beschikbare powerdata.
4. **Luisterkwaliteit** — alle WAV-bestanden blijven raw beschikbaar voor latere blinde menselijke beoordeling; esthetische kwaliteit wordt niet tot één willekeurige automatische score teruggebracht.

CLAP-evaluatietijd telt **niet** mee als generatiesnelheid.

## Praktijktests

| Test | Inhoud | Primaire meting |
|---|---|---|
| M1 | rustige solo-piano / eenvoudige genre-prompt | CLAP + menselijke review |
| M2 | cinematic orchestra met opgegeven instrumenten en mood | CLAP + menselijke review |
| M3 | 120 BPM electronic groove | CLAP + BPM-fout/tolerantie |
| M4 | melody-conditioned arrangement | capability + chroma-similarity proxy + luisterreview |
| M5 | langere intro → build → climax → resolution | coherence / raw luisterreview + performance |
| M6 | complexe prompt met A minor, ~105 BPM en meerdere instrumenten | CLAP + tempo/key-indicatoren |
| M7 | moderne jazztrio, gericht op blinde kwaliteitsvergelijking | raw WAV + CLAP |

M4 wordt alleen uitgevoerd voor modellen met de capability `melody`. Een text-only model krijgt hiervoor `unsupported`, niet `FAIL`.

## Externe referentielaag: MusicBench

De benchmark gebruikt `amaai-lab/MusicBench`, `MusicBench_test_B.json`, op de vaste revision:

```text
b141e962aacc19ffd51c15732738040377989203
```

MusicBench breidt MusicCaps uit met onder andere tempo, key, beat- en chordmetadata en bevat 400 testsamples. AI Benchmark v4 downloadt **alleen de test-B JSON metadata/prompts**, niet het 16,8 GB audio-archief. Voor iedere geselecteerde prompt wordt lokaal 10 seconden nieuwe audio gegenereerd.

| Profiel | MusicBench-prompts |
|---|---:|
| `quick` | 5 |
| `standard` | 20 |
| `full` | 400 |

Per gegenereerde MusicBench-output rapporteren we:

- CLAP text/audio cosine similarity;
- geschatte BPM en absolute fout t.o.v. MusicBench;
- aandeel binnen ±10 BPM;
- geschatte key en match met de MusicBench-key.

De tempo- en keydetectie zijn **lokale signal-analysis proxies** (librosa/chroma), geen officiële Mustango/MusicBench leaderboardscore. Ze worden daarom ook zo gelabeld in raw/summary.

## Vaste CLAP-evaluator

Promptmatching gebruikt:

```text
laion/clap-htsat-fused
revision 365dea6ef167def6676140ed93bbc43f84dabb28
```

De evaluator draait bewust op **CPU/float32**, ook als de generatie op CUDA of Metal draait. Daarmee blijft de kwaliteitsscore zo veel mogelijk onafhankelijk van de accelerator van de worker. Voor audio langer dan 10 seconden wordt deterministisch de eerste 10 seconden geëvalueerd, passend bij de CLAP-preprocessor.

## Generation backend

De eerste universele backend is Hugging Face Transformers MusicGen.

Standaard ingeschakeld:

```text
facebook/musicgen-small
revision f37b0c0576a5fb6891df4a25c80680af72c11e1e
```

De standaardconfig gebruikt `guidance_scale=3`, `top_k=250`, `temperature=1.0` en seed `42`.

Devicekeuze:

- Linux/WSL + CUDA-capabele PyTorch: CUDA/float16;
- macOS Apple Silicon: MPS/float32;
- anders: CPU/float32.

Optionele grotere of melody-capable MusicGen-modellen kunnen per machine in Git worden aangezet. Hun exacte Hugging Face revision wordt tijdens preflight resolved en in de preparation/raw metadata opgeslagen.

## Modelconfig per machine

Musicmodellen staan los van Ollama-, image- en speechmodellen:

```text
config/music-models/<machine>.models.tsv
```

Kolommen:

```text
enabled  backend  model  revision  capabilities  guidance  top_k  temperature  notes
```

Voorbeeld:

```text
true   transformers-musicgen  facebook/musicgen-small   <revision>  text         3.0  250  1.0  baseline
false  transformers-musicgen  facebook/musicgen-melody  main        text,melody  3.0  250  1.0  optional
```

De daadwerkelijke bestanden zijn tab-separated. De gekozen musicconfig wordt bij start in `run/config/music.models.tsv` gesnapshot, zodat `resume` niet ongemerkt een later gewijzigde Git-config gebruikt.

## Profielen

| Profiel | Praktijktests | MusicBench |
|---|---|---:|
| `quick` | M1, M3, M6 | 5 |
| `standard` | M1–M7 | 20 |
| `full` | M1–M7 | 400 |

## Raw output

Per model blijven de volledige WAV's en alle afgeleide gegevens beschikbaar:

```text
raw/music/<model>/
├── audio/
│   ├── practical/
│   └── musicbench/
├── practical/
└── external/
    └── musicbench/
```

Raw JSON bevat prompt, seed, exacte modelrevision, device/dtype, requested en werkelijke audio-duration, generation wall time, RTF, powerdata, VRAM/RAM waar beschikbaar, CLAP-score, BPM/keyanalyse en alle human-reviewvelden.

## Dependencies

`benchmarks/music/dependencies.sh` draait tijdens attached preflight en installeert automatisch PyTorch/Transformers, librosa, soundfile, Hugging Face tooling, de ingeschakelde modelweights, de vaste CLAP-evaluator en MusicBench testmetadata. Daarna heeft de headless generation-run geen downloads of sudo-prompts meer nodig.

## Licenties

Modelweights en MusicBench-data worden niet in deze Git-repository herverdeeld. De standaard `facebook/musicgen-small` checkpoint is upstream `CC-BY-NC-4.0`; MusicBench is `CC-BY-SA-3.0`; LAION CLAP is `Apache-2.0`. Zie `THIRD_PARTY.md` voor details.
