# Speech & Audio

Benchmark 6 meet lokale **speech-to-text (STT)**, **speaker diarization** en **text-to-speech (TTS)**. De module draait volledig lokaal nadat de attached preflight alle dependencies, modellen en vaste testaudio heeft gecachet.

## Doel

De benchmark houdt drie dingen uit elkaar:

1. **Transcriptiekwaliteit** — WER/CER op Nederlands en technische termen.
2. **Sprekerscheiding** — diarization error rate (DER) op een vaste tweesprekerfixture.
3. **Spraaksynthese** — snelheid, outputduur en verstaanbaarheid; de gegenereerde WAV blijft beschikbaar voor blinde menselijke vergelijking.

Performance wordt gerapporteerd als wall time en **real-time factor (RTF)**. Een RTF van `0.25` betekent dat 60 seconden audio in ongeveer 15 seconden wordt verwerkt/gegeneerd.

## STT praktijktests

| Test | Inhoud | Primaire meting |
|---|---|---|
| S1 | Schone Nederlandse FLEURS-opname | WER, CER, RTF |
| S2 | Nederlands met deterministische 10 dB achtergrondruis | WER, CER, robuustheid |
| S3 | Laravel/Livewire/Web Awesome/Redis in Nederlands | WER, technische termen |
| S4 | Vier afwisselende beurten met twee Nederlandse stemmen | WER + DER |
| S5 | Langere samengestelde Nederlandse opname | WER, stabiliteit, RTF |

S4 gebruikt bekende referentie-intervallen en een vaste lokale sherpa-onnx diarizationpipeline. De score gebruikt frame-based DER met de beste labelmapping tussen de twee anonieme speakers.

## Externe STT-referentie

De externe laag gebruikt `google/fleurs`, configuratie `nl_nl`, test split, vastgepind op revision:

```text
73c36572c7f01dea15fe27266e26c29f4cda9a83
```

De selectie is deterministisch en wordt tijdens preflight als 16 kHz mono PCM-WAV in de persistente cache gezet.

| Profiel | FLEURS NL testclips |
|---|---:|
| `quick` | 5 |
| `standard` | 20 |
| `full` | 350 |

## TTS praktijktests

| Test | Inhoud | Meting |
|---|---|---|
| T1 | Natuurlijke Nederlandse zin | RTF + verstaanbaarheid |
| T2 | Technische termen/merknamen | RTF + back-transcription WER |
| T3 | Langere rolling-deploymenttekst | RTF + consistentie |
| T4 | Vaste snelheidstekst | RTF / audio-seconden per seconde |

Iedere TTS-output wordt daarnaast door de **vaste Whisper small evaluator** teruggetranscribeerd. De WER daarvan is een reproduceerbare intelligibility-proxy, geen vervanging voor menselijke beoordeling van natuurlijkheid.

## Backends

### STT

STT gebruikt een vastgepinde build van `whisper.cpp`:

```text
v1.8.7
```

De exacte Git commit wordt in `prepared.json` en daarmee in raw metadata vastgelegd.

- macOS: Metal build
- Linux/WSL met NVIDIA + `nvcc`: CUDA build
- anders: CPU build

Een CUDA toolkit is **optionele hardware-acceleratie**, geen vereiste om de benchmark te kunnen draaien. Ontbreekt `nvcc`, dan meldt preflight dit expliciet en draait STT op CPU in plaats van de benchmark te blokkeren.

### TTS en diarization

TTS en speaker diarization gebruiken `sherpa-onnx==1.13.7`. De standaard Nederlandse TTS-baseline is:

```text
vits-piper-nl_NL-alex-medium
```

De tweede Nederlandse stem `vits-piper-nl_NL-pim-medium` wordt gebruikt om de vaste S4-tweesprekerfixture te maken en kan ook als zelfstandig TTS-model worden ingeschakeld.

Providerkeuze voor TTS is best effort: CoreML op macOS, CUDA op een NVIDIA-worker wanneer de geïnstalleerde runtime dit ondersteunt, anders CPU. De daadwerkelijk gebruikte provider wordt per model opgeslagen.

## Modelconfig per machine

Speechmodellen staan los van Ollama- en image-modellen:

```text
config/speech-models/<machine>.models.tsv
```

Voorbeeld:

```text
true  stt  whisper-small  small  nl
true  tts  piper-nl-alex  vits-piper-nl_NL-alex-medium  nl_NL  0
```

Je kunt zwaardere Whispermodellen per machine in Git activeren zonder andere machines te wijzigen.

## Profielen

| Profiel | Practical STT | FLEURS | TTS |
|---|---:|---:|---:|
| `quick` | S1–S3 | 5 | T1, T2, T4 |
| `standard` | S1–S5 | 20 | T1–T4 |
| `full` | S1–S5 | 350 | T1–T4 |

Diarization S4 wordt eenmaal per Speech-run uitgevoerd, omdat daarvoor bewust één vaste benchmarkpipeline wordt gebruikt en niet ieder STT-model een eigen speakerseparator krijgt.

## Raw output

De raw map bewaart onder meer:

```text
raw/speech/
├── diarization/
├── stt/
│   ├── practical/
│   └── fleurs/
└── tts/
```

Daarin blijven transcripties, referentietekst, WER/CER, whisper.cpp stdout/stderr/timings, power samples, diarizationsegmenten, TTS WAV-bestanden en back-transcripties beschikbaar. De compacte `summary.json` wordt uitsluitend uit deze ruwe resultaten afgeleid.

## Dependencies

`benchmarks/speech/dependencies.sh` wordt tijdens attached preflight uitgevoerd en installeert automatisch de vereiste systeem- en Pythonpackages, bouwt whisper.cpp, downloadt de ingeschakelde modellen en prepareert alle audiofixtures. Daardoor vraagt de latere headless job niet alsnog om sudo of downloads.
