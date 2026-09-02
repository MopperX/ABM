#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.benchlib import parse_model_rows

WHISPER_CPP_TAG = "v1.8.7"
FLEURS_REV = "73c36572c7f01dea15fe27266e26c29f4cda9a83"
SHERPA_VERSION = "1.13.7"
TTS_ASSETS = {
    "vits-piper-nl_NL-alex-medium": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-nl_NL-alex-medium.tar.bz2",
    "vits-piper-nl_NL-pim-medium": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-nl_NL-pim-medium.tar.bz2",
}
DIAR_SEG_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
DIAR_EMB_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
PRACTICAL_BY_PROFILE = {
    "quick": ["S1", "S2", "S3"],
    "standard": ["S1", "S2", "S3", "S4", "S5"],
    "full": ["S1", "S2", "S3", "S4", "S5"],
}
FLEURS_COUNTS = {"quick": 5, "standard": 20, "full": 350}


def atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def speech_config_for(machine_cfg: Path, repo: Path) -> Path:
    return machine_cfg


def parse_models(path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in parse_model_rows(path):
        suites={x.strip() for x in row['suites'].split(',') if x.strip()}
        backend=row['backend'].lower()
        if 'speech' not in suites or backend not in {'whispercpp','sherpa-onnx-tts'}:
            continue
        kind='stt' if backend == 'whispercpp' else 'tts'
        model=row['model']
        out.append({
            'enabled':row['enabled'], 'kind':kind, 'id':model, 'model':model,
            'language':row['language'], 'speaker':row['speaker'], 'notes':row['notes'],
        })
    return out

def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloaden: {url}", flush=True)
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def extract_tar_bz2(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(dest)


def ensure_whisper_cpp(cache: Path, stt_models: list[str]) -> dict[str, Any]:
    root = cache / "whisper.cpp"
    src = root / "src"
    if not (src / ".git").exists():
        root.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--quiet", "--branch", WHISPER_CPP_TAG, "--depth", "1", "https://github.com/ggml-org/whisper.cpp.git", str(src)])
    else:
        run(["git", "-C", str(src), "fetch", "--quiet", "--tags", "origin"])
        run(["git", "-C", str(src), "checkout", "--quiet", WHISPER_CPP_TAG])

    build = src / "build"
    cmake_args = ["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
    backend = "cpu"
    if platform.system() == "Darwin":
        cmake_args += ["-DGGML_METAL=ON"]
        backend = "metal"
    elif shutil.which("nvidia-smi") and shutil.which("nvcc"):
        cmake_args += ["-DGGML_CUDA=ON"]
        backend = "cuda"
    elif shutil.which("nvidia-smi"):
        print("WAARSCHUWING: NVIDIA GPU gevonden maar nvcc ontbreekt; whisper.cpp gebruikt CPU. Installeer een CUDA toolkit om STT-CUDA te benchmarken.", flush=True)
    run(cmake_args)
    run(["cmake", "--build", str(build), "--config", "Release", "-j", str(max(1, min(os.cpu_count() or 2, 12)))])
    cli = build / "bin" / "whisper-cli"
    if not cli.exists():
        raise RuntimeError(f"whisper-cli niet gevonden na build: {cli}")

    model_dir = src / "models"
    model_paths: dict[str, str] = {}
    # small is also used as fixed TTS intelligibility evaluator.
    needed = list(dict.fromkeys(stt_models + ["small"]))
    for model in needed:
        path = model_dir / f"ggml-{model}.bin"
        if not path.exists():
            run(["bash", str(model_dir / "download-ggml-model.sh"), model])
        model_paths[model] = str(path)
    revision = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    return {"tag": WHISPER_CPP_TAG, "revision": revision, "backend": backend, "cli": str(cli), "models": model_paths}


def ensure_tts_assets(cache: Path, enabled_tts: list[str]) -> dict[str, Any]:
    root = cache / "tts-models"
    # Alex + Pim are always needed to construct the fixed practical S3/S4 fixtures.
    needed = list(dict.fromkeys(enabled_tts + ["vits-piper-nl_NL-alex-medium", "vits-piper-nl_NL-pim-medium"]))
    out: dict[str, Any] = {}
    for model in needed:
        url = TTS_ASSETS.get(model)
        if not url:
            raise RuntimeError(f"Onbekend TTS-model in speech config: {model}")
        archive = root / f"{model}.tar.bz2"
        model_dir = root / model
        if not model_dir.exists():
            download(url, archive)
            extract_tar_bz2(archive, root)
        onnx = next(model_dir.glob("*.onnx"), None)
        tokens = model_dir / "tokens.txt"
        data_dir = model_dir / "espeak-ng-data"
        if not onnx or not tokens.exists() or not data_dir.exists():
            raise RuntimeError(f"Onvolledige sherpa/Piper modelmap: {model_dir}")
        out[model] = {"dir": str(model_dir), "onnx": str(onnx), "tokens": str(tokens), "data_dir": str(data_dir), "url": url}
    return out


def ensure_diarization(cache: Path) -> dict[str, str]:
    root = cache / "diarization"
    seg_archive = root / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    seg_dir = root / "sherpa-onnx-pyannote-segmentation-3-0"
    if not seg_dir.exists():
        download(DIAR_SEG_URL, seg_archive)
        extract_tar_bz2(seg_archive, root)
    emb = root / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    download(DIAR_EMB_URL, emb)
    seg = seg_dir / "model.onnx"
    if not seg.exists():
        seg = seg_dir / "model.int8.onnx"
    return {"segmentation": str(seg), "embedding": str(emb)}


def even_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    if count <= 1:
        return [0]
    return sorted(set(round(i * (total - 1) / (count - 1)) for i in range(count)))


def _write_audio_bytes(audio_obj: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = audio_obj.get("bytes")
    path = audio_obj.get("path")
    with tempfile.TemporaryDirectory() as td:
        source = Path(td) / (Path(path or "source.wav").name or "source.wav")
        if raw:
            source.write_bytes(raw)
        elif path and Path(path).exists():
            shutil.copy2(path, source)
        else:
            raise RuntimeError("FLEURS audio bevat geen bruikbare bytes/path")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)])


def prepare_fleurs(profile: str, cache: Path) -> dict[str, Any]:
    out_dir = cache / "fleurs" / profile
    manifest_path = out_dir / "manifest.json"
    desired = FLEURS_COUNTS[profile]
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if m.get("revision") == FLEURS_REV and m.get("count") == desired and all(Path(x["wav"]).exists() for x in m.get("samples", [])):
            return m

    from datasets import Audio, load_dataset
    ds = load_dataset("google/fleurs", "nl_nl", split="test", streaming=True, revision=FLEURS_REV)
    ds = ds.cast_column("audio", Audio(decode=False))
    rows = list(ds)
    idxs = even_indices(len(rows), desired)
    samples = []
    for pos, idx in enumerate(idxs):
        row = rows[idx]
        wav = out_dir / "audio" / f"{pos:04d}-source-{idx}.wav"
        if not wav.exists():
            _write_audio_bytes(row["audio"], wav)
        samples.append({
            "id": f"fleurs-nl-{idx}", "source_index": idx, "wav": str(wav),
            "transcription": str(row.get("transcription") or row.get("raw_transcription") or "").strip(),
            "raw_transcription": str(row.get("raw_transcription") or "").strip(),
            "gender": str(row.get("gender") or ""),
        })
    m = {"source": "google/fleurs", "config": "nl_nl", "split": "test", "revision": FLEURS_REV, "profile": profile, "count": len(samples), "samples": samples}
    atomic(manifest_path, m)
    return m


def tts_provider() -> str:
    if platform.system() == "Darwin":
        return "coreml"
    if shutil.which("nvidia-smi"):
        # sherpa-onnx may still fall back/fail if the pip wheel lacks CUDA; synthesis helper retries CPU.
        return "cuda"
    return "cpu"


def synth(model: dict[str, str], text: str, out: Path, sid: int = 0, provider: str | None = None) -> dict[str, Any]:
    import sherpa_onnx
    provider = provider or tts_provider()

    def make(p: str):
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=model["onnx"], tokens=model["tokens"], data_dir=model["data_dir"]),
                provider=p, debug=False, num_threads=max(1, min(os.cpu_count() or 2, 8)),
            ),
            max_num_sentences=1,
        )
        if not cfg.validate():
            raise RuntimeError(f"Ongeldige sherpa TTS-config voor provider {p}")
        return sherpa_onnx.OfflineTts(cfg)

    used = provider
    try:
        tts = make(provider)
    except Exception:
        if provider == "cpu":
            raise
        used = "cpu"
        tts = make("cpu")
    gen_cfg = sherpa_onnx.GenerationConfig(); gen_cfg.sid = sid; gen_cfg.speed = 1.0; gen_cfg.silence_scale = 0.2
    audio = tts.generate(text, gen_cfg)
    if len(audio.samples) == 0:
        raise RuntimeError("TTS genereerde geen samples")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio.samples, audio.sample_rate, subtype="PCM_16")
    return {"provider": used, "sample_rate": int(audio.sample_rate), "samples": len(audio.samples), "duration_s": len(audio.samples) / float(audio.sample_rate)}


def read_mono(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "x.wav"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(target_sr), "-c:a", "pcm_f32le", str(tmp)])
        x, sr = sf.read(tmp, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return np.asarray(x, dtype=np.float32), int(sr)


def write_wav(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -0.99, 0.99), sr, subtype="PCM_16")


def prepare_practical(repo: Path, cache: Path, fleurs: dict[str, Any], tts_assets: dict[str, Any]) -> dict[str, Any]:
    manifest_path = cache / "practical" / "manifest.json"
    tests = json.loads((repo / "benchmarks/speech/fixtures/practical/tests.json").read_text(encoding="utf-8"))
    stt_defs = {x["id"]: x for x in tests["stt"]}
    pdir = cache / "practical" / "audio"
    samples = fleurs["samples"]
    if len(samples) < 5:
        raise RuntimeError("Onvoldoende FLEURS-samples voor praktische speech-fixtures")

    entries: dict[str, Any] = {}
    # S1 clean FLEURS.
    s1 = pdir / "S1-clean.wav"; shutil.copy2(samples[0]["wav"], s1)
    entries["S1"] = {"wav": str(s1), "reference": samples[0]["transcription"], "source": samples[0]["id"]}

    # S2 deterministic white-noise version at 10 dB SNR.
    clean, sr = read_mono(Path(samples[1]["wav"]))
    rng = np.random.default_rng(42); noise = rng.normal(0, 1, len(clean)).astype(np.float32)
    sig_rms = math.sqrt(float(np.mean(clean ** 2)) + 1e-12); noise_rms = math.sqrt(float(np.mean(noise ** 2)) + 1e-12)
    target_noise = sig_rms / (10 ** (10 / 20)); noisy = clean + noise * (target_noise / noise_rms)
    s2 = pdir / "S2-noise-10db.wav"; write_wav(s2, noisy, sr)
    entries["S2"] = {"wav": str(s2), "reference": samples[1]["transcription"], "source": samples[1]["id"], "snr_db": 10}

    # S3 technical synthetic, standardized source voice.
    s3 = pdir / "S3-technical.wav"
    info3 = synth(tts_assets["vits-piper-nl_NL-alex-medium"], stt_defs["S3"]["reference"], s3)
    x3, _ = read_mono(s3); write_wav(s3, x3, 16000)
    entries["S3"] = {"wav": str(s3), "reference": stt_defs["S3"]["reference"], "synthesis": info3}

    # S4 two speakers, alternating fixed turns with known reference intervals.
    turns = stt_defs["S4"]["turns"]; pieces=[]; intervals=[]; cursor=0; silence=np.zeros(int(0.65*16000), dtype=np.float32)
    for i, text in enumerate(turns):
        voice = "vits-piper-nl_NL-alex-medium" if i % 2 == 0 else "vits-piper-nl_NL-pim-medium"
        tmp = pdir / f"S4-turn-{i+1}.wav"; synth(tts_assets[voice], text, tmp); x, _ = read_mono(tmp)
        start = cursor / 16000.0; pieces.append(x); cursor += len(x); end = cursor / 16000.0
        intervals.append({"start": start, "end": end, "speaker": i % 2, "text": text, "voice": voice})
        if i != len(turns)-1:
            pieces.append(silence); cursor += len(silence)
    s4 = pdir / "S4-two-speaker.wav"; write_wav(s4, np.concatenate(pieces), 16000)
    entries["S4"] = {"wav": str(s4), "reference": " ".join(turns), "turns": intervals, "speaker_count": 2}

    # S5 long-ish real Dutch speech, using clean FLEURS utterances with short pauses.
    longs=[]; refs=[]; sil=np.zeros(int(.35*16000), dtype=np.float32)
    long_samples = samples[:min(10, len(samples))]
    for i, row in enumerate(long_samples):
        x,_=read_mono(Path(row["wav"])); longs.append(x); refs.append(row["transcription"])
        if i != len(long_samples)-1: longs.append(sil)
    s5=pdir/"S5-long.wav"; write_wav(s5,np.concatenate(longs),16000)
    entries["S5"]={"wav":str(s5),"reference":" ".join(refs),"source_ids":[x["id"] for x in long_samples]}

    m={"schema_version":1,"entries":entries,"source_fleurs_revision":FLEURS_REV,"fixture_tts":"Alex/Pim Piper VITS via sherpa-onnx"}
    atomic(manifest_path,m); return m


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--profile",choices=["quick","standard","full"],required=True); ap.add_argument("--cache-root",required=True); ap.add_argument("--machine-config",required=True)
    a=ap.parse_args(); repo=Path(__file__).resolve().parents[1]; cache=Path(a.cache_root)/"speech"; cache.mkdir(parents=True,exist_ok=True)
    cfg=speech_config_for(Path(a.machine_config),repo); models=parse_models(cfg)
    stt=[m["model"] for m in models if m["kind"]=="stt"]; tts=[m["model"] for m in models if m["kind"]=="tts"]
    whisper=ensure_whisper_cpp(cache,stt); tts_assets=ensure_tts_assets(cache,tts); diar=ensure_diarization(cache); fleurs=prepare_fleurs(a.profile,cache); practical=prepare_practical(repo,cache,fleurs,tts_assets)
    atomic(cache/"prepared.json",{
        "whisper_cpp":whisper,"sherpa_onnx_version":SHERPA_VERSION,"speech_model_config":str(cfg),"models":models,
        "tts_assets":tts_assets,"diarization":diar,"fleurs_manifest":str(cache/"fleurs"/a.profile/"manifest.json"),"practical_manifest":str(cache/"practical"/"manifest.json"),
    })
    print(f"Speech preflight gereed: {len(stt)} STT-model(len), {len(tts)} TTS-model(len), FLEURS {len(fleurs['samples'])} samples.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
