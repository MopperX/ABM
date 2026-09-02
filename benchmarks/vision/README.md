# Vision / Screenshots

Vision meet praktische screenshotanalyse én externe multimodale referentietests. De benchmark gebruikt Ollama vision-input via base64-afbeeldingen. Hybride modellen worden per reasoning-mode gescheiden; modellen die aparte thinking/instruct-gewichten publiceren worden als aparte officiële modeltags getest.

## Praktijktests

- **V1** technische foutpagina: feiten, directe foutoorzaak en veilige volgende controle
- **V2** UI-layoutprobleem: zichtbare clipping/overflow herkennen
- **V3** screenshot + Blade/Livewire-code combineren
- **V4** exacte OCR van technische logregels
- **V5** GUI-grounding: klikpunt moet binnen een bekende bounding box vallen
- **V6** technisch diagram begrijpen
- **V7** epistemische terughoudendheid: geen onzichtbare oorzaak verzinnen

Profielen:

| Profiel | Praktijktests | Repeats |
|---|---|---:|
| quick | V1, V4, V5, V7 | 1 |
| standard | V1–V7 | 3 |
| full | V1–V7 | 3 |

## Externe laag

### ScreenSpot

GUI-grounding met echte screenshots en geannoteerde target-bounding-boxes. Het model retourneert een pixelcoördinaat; de benchmark rekent dit waar nodig terug naar de genormaliseerde ScreenSpot-coördinaten en rapporteert click accuracy.

- quick: 6 deterministisch geselecteerde samples
- standard: 30 deterministisch geselecteerde samples
- full: volledige testset

### MMMU-Pro

De `standard (10 options)`-configuratie meet bredere multimodale reasoning. Alleen de modelkeuze A–J wordt gescoord; raw antwoord, thinking en timings blijven opgeslagen.

- quick: 14 deterministisch geselecteerde vragen
- standard: 42 deterministisch geselecteerde vragen
- full: volledige testset

## Datasetcache

De eerste Vision-preflight downloadt de datasets naar de persistente benchmarkcache buiten Git. De eerste aangetroffen Hugging Face commit-SHA wordt in `cache/vision/sources.json` bevroren en bij latere runs opnieuw gebruikt. Hierdoor verandert een publieke datasetupdate niet stilletjes de benchmarkbasis.

De geselecteerde afbeeldingen plus manifesten worden daarnaast onder `cache/vision/screenspot/<profile>/` en `cache/vision/mmmu-pro/<profile>/` vastgelegd.

## Raw output

Per model/reasoningmodus worden onder andere opgeslagen:

- volledige Ollama request/response
- `message.thinking` wanneer beschikbaar
- oorspronkelijke fixture/dataset-ID
- afbeelding en ground truth metadata
- clickpunt/bounding-box voor grounding
- antwoordletter/gold label voor MMMU-Pro
- timings/tok/s
- beschikbare power-sampling

De praktische fixtures in deze repository zijn synthetisch en doelbewust klein. De externe datasets worden niet in Git opgenomen.

## Qwen3-VL baseline

Qwen3-VL publiceert aparte Instruct- en Thinking-tags. Daarom gebruikt de standaard machineconfig zowel `qwen3-vl:4b-instruct` als `qwen3-vl:4b-thinking`, ieder met `mode=standard`. We simuleren hier geen no-thinking door `think:false` op de Thinking-tag te forceren. Voor hybride modellen zoals die in andere suites voorkomen blijven `thinking` en `nothinking` gewone reasoning-modi.
