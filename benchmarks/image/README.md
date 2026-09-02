# Image Generation

De Image Generation-module meet **generatiesnelheid, resourcegebruik, promptvolging en visuele voorkeur afzonderlijk**. Er wordt geen enkel magisch totaalcijfer gemaakt. Alle PNG-bestanden blijven in `raw/` beschikbaar voor latere blinde menselijke vergelijking.

## Modellen per machine

Image-modellen worden apart van Ollama beheerd in `config/image-models/<machine>.models.tsv`. Daardoor kun je in de publieke Git per machine eenvoudig modellen aan/uit zetten zonder de LLM-config te vervuilen.

Kolommen: `enabled`, Hugging Face `model`, vaste `revision`, `steps`, `guidance`, `offload`, `notes`.

Standaard staat alleen `segmind/tiny-sd` aan als lichte universele baseline. `sd-turbo` staat klaar als zwaardere optie. FLUX.1-schnell staat als **gated** voorbeeld uit; als je die activeert moet de Hugging Face-licentie vooraf handmatig geaccepteerd zijn en moet een token beschikbaar zijn.

## Praktijktests I1-I7

- I1 eenvoudig object
- I2 objecten + ruimtelijke relatie
- I3 tellen, kleur en positie
- I4 tekst in afbeelding
- I5 fotorealisme
- I6 illustratie/concept art
- I7 technische/UI-illustratie

I1-I4 krijgen daarnaast een vaste, gemeenschappelijke Vision-evaluator (`qwen3-vl:4b-instruct`) voor machine-checkbare promptcriteria. I5-I7 worden bewust niet tot één automatische 'mooicijfer'-score gereduceerd; de outputs worden opgeslagen voor menselijke/blinde beoordeling.

## Externe referentielagen

### GenEval2-compatible local VQA

Preflight haalt de officiële GenEval2 promptdata op op een vaste revision. Quick gebruikt 5 vaste prompts, standard 20 en full alle 800. De officiële prompts/atom-vragen en ground truth worden gebruikt. Voor universele lokale uitvoering beantwoorden we de VQA-atomen met dezelfde vaste Ollama Vision-evaluator. Dit resultaat heet daarom **GenEval2 local-VQA atom accuracy** en is nadrukkelijk niet hetzelfde als de gepubliceerde officiële Soft-TIFA-score. Raw data bewaart de officiële prompt, atomen, judge-output en bronrevision.

### HPS v2.1

De officiële `hpsv2` 1.2.0-package wordt gebruikt. Quick genereert 1 prompt per HPS-stijl, standard 5 per stijl; full gebruikt alle beschikbare benchmarkprompts. De geproduceerde beelden worden met HPS v2.1 gescoord. HPS is vooral bedoeld om beelden voor dezelfde prompt te vergelijken; de raw per-promptscores blijven daarom altijd beschikbaar naast gemiddelden.

## Resolutie en seeds

De vergelijkbare hoofdset draait op **512×512**. Iedere prompt krijgt een deterministische seed afgeleid van de algemene benchmarkseed, zodat verschillende machines voor dezelfde modelconfig dezelfde seeds krijgen. Model-specifieke aanbevolen `steps` en `guidance` staan in de Git-config.

## Performance

Per model worden opgeslagen:

- model load time;
- generation wall time per afbeelding;
- images/minute;
- NVIDIA power/energie indien meetbaar;
- CUDA peak allocated/reserved VRAM;
- MPS allocated memory waar beschikbaar;
- process max RSS;
- output SHA-256;
- HPS/judge-tijd apart van generation time.

Een OOM of niet-ondersteund model wordt als `unsupported`/`error` vastgelegd en niet als slechte beeldkwaliteit meegeteld.

## Profiles

- `quick`: I1, I3, I4 + 5 GenEval2 + 4 HPS-prompts.
- `standard`: I1-I7 + 20 GenEval2 + 20 HPS-prompts.
- `full`: I1-I7 + volledige GenEval2 + alle HPS benchmarkprompts.

## Dependencies

`benchmarks/image/dependencies.sh` installeert PyTorch/Diffusers/HPSv2 en downloadt alle **ingeschakelde** image-modellen tijdens preflight, dus vóór de headless job wordt losgekoppeld. Model- en evaluatordata komen in de persistente benchmarkcache buiten Git.
