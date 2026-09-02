from __future__ import annotations

import csv
import gc
import json
import math
import os
import resource
import statistics
import time
from pathlib import Path
from typing import Any


from lib.benchlib import PowerSampler, atomic_json, load_json, parse_model_rows, utc_now

PRACTICAL = {
    "quick": ["M1", "M3", "M6"],
    "standard": ["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
    "full": ["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
}
SEED = 42


def parse_music_models(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for row in parse_model_rows(path):
        suites={x.strip() for x in row['suites'].split(',') if x.strip()}
        if 'music' not in suites or row['backend'].lower() != 'transformers-musicgen':
            continue
        out.append({
            'backend':row['backend'], 'model':row['model'], 'revision':row['revision'] or 'main',
            'capabilities':{x.strip() for x in (row['capabilities'] or 'text').split(',') if x.strip()},
            'guidance':float(row['guidance'] or 3.0), 'top_k':int(row['top_k'] or 250),
            'temperature':float(row['temperature'] or 1.0), 'notes':row['notes'],
        })
    return out

def _cache_root(run_dir: Path) -> Path:
    return run_dir.parents[2] / "cache" / "music"


def _prepared(run_dir: Path) -> dict[str, Any]:
    p = _cache_root(run_dir) / "prepared.json"
    if not p.exists():
        raise RuntimeError("Music preflight manifest ontbreekt")
    return load_json(p)


def job_count(run_dir: Path, profile: str, music_config: Path) -> int:
    prep = _prepared(run_dir)
    mb = load_json(Path(prep["musicbench_manifest"]))
    return len(parse_music_models(music_config)) * (len(PRACTICAL[profile]) + int(mb.get("count", 0)))


def _rss_mb() -> float:
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1024.0 if os.uname().sysname != "Darwin" else v / (1024.0 * 1024.0)


def _device() -> tuple[str, Any]:
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps", torch.float32
    return "cpu", torch.float32


def _to_device(batch: Any, device: str) -> dict[str, Any]:
    import torch

    out = {}
    for k, v in dict(batch).items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class MusicGenGenerator:
    def __init__(self, prepared: dict[str, Any], cfg: dict[str, Any]):
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenMelodyForConditionalGeneration

        self.cfg = cfg
        self.prepared = prepared
        self.device, self.dtype = _device()
        self.local_path = prepared["local_path"]
        self.resolved_revision = prepared.get("resolved_revision")
        started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(self.local_path, local_files_only=True)
        model_cls = MusicgenMelodyForConditionalGeneration if "melody" in cfg.get("capabilities", set()) else MusicgenForConditionalGeneration
        self.model = model_cls.from_pretrained(
            self.local_path,
            local_files_only=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.load_seconds = time.monotonic() - started
        self.sample_rate = int(getattr(self.model.config.audio_encoder, "sampling_rate", 32000) or 32000)
        self.frame_rate = float(getattr(self.model.config.audio_encoder, "frame_rate", 50.0) or 50.0)
        self.torch = torch

    def generate(self, *, prompt: str, duration: float, output: Path, melody: Path | None = None, seed: int = SEED) -> dict[str, Any]:
        import librosa
        import numpy as np
        import soundfile as sf

        torch = self.torch
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        before_rss = _rss_mb()

        if melody is not None:
            audio, sr = sf.read(melody, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            target_sr = int(getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", sr) or sr)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            inputs = self.processor(audio=audio, sampling_rate=sr, text=[prompt], padding=True, truncation=True, return_tensors="pt")
        else:
            inputs = self.processor(text=[prompt], padding=True, truncation=True, return_tensors="pt")
        inputs = _to_device(inputs, self.device)

        max_new_tokens = max(1, int(round(duration * self.frame_rate)))
        sampler = PowerSampler(interval=0.5)
        sampler.start()
        started = time.monotonic()
        ok = True
        err = None
        values = None
        try:
            with torch.inference_mode():
                values = self.model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=self.cfg["guidance"],
                    top_k=self.cfg["top_k"],
                    temperature=self.cfg["temperature"],
                    max_new_tokens=max_new_tokens,
                )
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {e}"
        elapsed = time.monotonic() - started
        power = sampler.stop(elapsed)

        peak_vram = None
        mps_allocated = None
        if self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        elif self.device == "mps" and hasattr(torch, "mps"):
            try:
                mps_allocated = torch.mps.current_allocated_memory() / (1024**2)
            except Exception:
                pass

        meta: dict[str, Any] = {
            "ok": ok,
            "error": err,
            "seconds": elapsed,
            "requested_duration_seconds": duration,
            "device": self.device,
            "dtype": str(self.dtype),
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "guidance_scale": self.cfg["guidance"],
            "top_k": self.cfg["top_k"],
            "temperature": self.cfg["temperature"],
            "power": power,
            "peak_cuda_vram_mb": peak_vram,
            "mps_allocated_mb_after": mps_allocated,
            "rss_mb_before": before_rss,
            "rss_mb_after": _rss_mb(),
        }
        if not ok or values is None:
            return meta

        arr = values.detach().to("cpu", dtype=torch.float32).numpy()
        # Transformers MusicGen returns [batch, channels, samples].
        if arr.ndim == 3:
            arr = arr[0]
            data = arr.T
        elif arr.ndim == 2:
            data = arr[0]
        else:
            data = arr.reshape(-1)
        data = np.asarray(data, dtype=np.float32)
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 1.0:
            data = data / peak
        sf.write(output, data, self.sample_rate, subtype="PCM_16")
        actual_frames = data.shape[0]
        actual_duration = actual_frames / self.sample_rate
        meta.update(
            {
                "audio": str(output),
                "sample_rate": self.sample_rate,
                "channels": int(data.shape[1]) if data.ndim == 2 else 1,
                "audio_frames": int(actual_frames),
                "audio_duration_seconds": actual_duration,
                "audio_seconds_per_generation_second": actual_duration / elapsed if elapsed > 0 else None,
                "real_time_factor": elapsed / actual_duration if actual_duration > 0 else None,
            }
        )
        return meta

    def close(self) -> None:
        try:
            del self.model
            del self.processor
        except Exception:
            pass
        gc.collect()
        try:
            if self.device == "cuda":
                self.torch.cuda.empty_cache()
            elif self.device == "mps" and hasattr(self.torch, "mps"):
                self.torch.mps.empty_cache()
        except Exception:
            pass


class ClapScorer:
    """Fixed CPU/float32 evaluator so quality metrics don't depend on accelerator availability."""

    def __init__(self, prep: dict[str, Any]):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.torch = torch
        local = prep["clap"]["local_path"]
        started = time.monotonic()
        self.processor = ClapProcessor.from_pretrained(local, local_files_only=True)
        self.model = ClapModel.from_pretrained(local, local_files_only=True, torch_dtype=torch.float32)
        self.model.to("cpu")
        self.model.eval()
        self.load_seconds = time.monotonic() - started

    def score(self, prompt: str, wav: Path) -> dict[str, Any]:
        import librosa
        import numpy as np
        import soundfile as sf
        import torch.nn.functional as F

        audio, sr = sf.read(wav, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # CLAP is configured for at most 10 s. Deterministically evaluate the first 10 s.
        max_seconds = 10.0
        if len(audio) > int(sr * max_seconds):
            audio = audio[: int(sr * max_seconds)]
        target_sr = int(getattr(self.processor.feature_extractor, "sampling_rate", 48000) or 48000)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        t0 = time.monotonic()
        text_inputs = self.processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
        audio_inputs = self.processor(audios=audio, sampling_rate=sr, return_tensors="pt")
        with self.torch.inference_mode():
            te = self.model.get_text_features(**text_inputs)
            ae = self.model.get_audio_features(**audio_inputs)
            score = F.cosine_similarity(te, ae, dim=-1).item()
        return {"cosine_similarity": float(score), "seconds": time.monotonic() - t0, "evaluated_seconds": len(audio) / sr}


def _norm_key(k: str | None) -> str | None:
    if not k:
        return None
    s = " ".join(str(k).replace("#", "#").split()).strip().lower()
    aliases = {"maj": "major", "min": "minor"}
    parts = s.split()
    if len(parts) >= 2 and parts[1] in aliases:
        parts[1] = aliases[parts[1]]
    return " ".join(parts[:2]) if parts else None


def _estimate_key(y: "np.ndarray", sr: int) -> dict[str, Any]:
    import librosa
    import numpy as np

    if y.size < 2048:
        return {"key": None, "correlation": None}
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    vec = np.nanmean(chroma, axis=1)
    if not np.any(np.isfinite(vec)) or np.linalg.norm(vec) == 0:
        return {"key": None, "correlation": None}
    vec = np.nan_to_num(vec)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best = (-999.0, None)
    for root in range(12):
        for mode, prof in (("major", major), ("minor", minor)):
            p = np.roll(prof, root)
            corr = float(np.corrcoef(vec, p)[0, 1])
            if math.isfinite(corr) and corr > best[0]:
                best = (corr, f"{names[root]} {mode}")
    return {"key": best[1], "correlation": best[0] if best[1] else None}


def audio_features(wav: Path) -> dict[str, Any]:
    import librosa
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(wav, dtype="float32", always_2d=True)
    channels = int(audio.shape[1])
    mono = np.mean(audio, axis=1)
    duration = len(mono) / sr
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    clipping = float(np.mean(np.abs(mono) >= 0.999)) if mono.size else 0.0
    try:
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
        tempo = float(np.asarray(tempo).reshape(-1)[0])
    except Exception:
        tempo = None
    key = _estimate_key(mono, sr)
    return {
        "sample_rate": int(sr),
        "channels": channels,
        "duration_seconds": duration,
        "rms": rms,
        "clipping_fraction": clipping,
        "estimated_bpm": tempo,
        "estimated_key": key["key"],
        "key_correlation": key["correlation"],
    }


def melody_chroma_similarity(reference: Path, generated: Path) -> float | None:
    import librosa
    import numpy as np
    import soundfile as sf

    try:
        a, sra = sf.read(reference, dtype="float32", always_2d=True)
        b, srb = sf.read(generated, dtype="float32", always_2d=True)
        a = np.mean(a, axis=1)
        b = np.mean(b, axis=1)
        ca = np.mean(librosa.feature.chroma_cqt(y=a, sr=sra), axis=1)
        cb = np.mean(librosa.feature.chroma_cqt(y=b, sr=srb), axis=1)
        den = float(np.linalg.norm(ca) * np.linalg.norm(cb))
        if den <= 0:
            return None
        return float(np.dot(ca, cb) / den)
    except Exception:
        return None


def _prepared_model(prep: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    for m in prep.get("models", []):
        if m.get("model") == cfg["model"] and (m.get("revision") or "main") == cfg["revision"]:
            return m
    return None


def _objective_checks(task: dict[str, Any], features: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if task.get("expected_bpm") is not None and features.get("estimated_bpm") is not None:
        err = abs(float(features["estimated_bpm"]) - float(task["expected_bpm"]))
        checks.append({"name": "tempo-within-tolerance", "passed": err <= float(task.get("bpm_tolerance", 10)), "error_bpm": err})
    if task.get("expected_key") and features.get("estimated_key"):
        checks.append({"name": "key-match-estimator", "passed": _norm_key(features["estimated_key"]) == _norm_key(task["expected_key"]), "estimated": features["estimated_key"], "expected": task["expected_key"]})
    return checks


def _write_result(path: Path, result: dict[str, Any]) -> None:
    atomic_json(path, result)


def _summary(cfg: dict[str, Any], generator: MusicGenGenerator | None, practical: list[dict[str, Any]], external: list[dict[str, Any]]) -> dict[str, Any]:
    allrows = practical + external
    times = [r.get("generation", {}).get("seconds") for r in allrows if r.get("generation", {}).get("ok")]
    audio_secs = [r.get("generation", {}).get("audio_duration_seconds") for r in allrows if r.get("generation", {}).get("ok")]
    clap = [r.get("clap", {}).get("cosine_similarity") for r in allrows if r.get("clap", {}).get("cosine_similarity") is not None]
    energy = [r.get("generation", {}).get("power", {}).get("approx_energy_wh") for r in allrows if r.get("generation", {}).get("power", {}).get("available")]
    bpm_err = [r.get("music_control", {}).get("bpm_absolute_error") for r in external if r.get("music_control", {}).get("bpm_absolute_error") is not None]
    bpm_ok = [r.get("music_control", {}).get("bpm_within_10") for r in external if r.get("music_control", {}).get("bpm_within_10") is not None]
    key_ok = [r.get("music_control", {}).get("key_match") for r in external if r.get("music_control", {}).get("key_match") is not None]
    return {
        "model": cfg["model"],
        "revision": cfg["revision"],
        "resolved_revision": getattr(generator, "resolved_revision", None),
        "backend": cfg["backend"],
        "capabilities": sorted(cfg["capabilities"]),
        "status": "completed",
        "device": getattr(generator, "device", None),
        "load_seconds": getattr(generator, "load_seconds", None),
        "practical": {
            "items": len(practical),
            "unsupported": sum(1 for r in practical if r.get("status") == "unsupported"),
            "clap_mean": statistics.mean([r["clap"]["cosine_similarity"] for r in practical if r.get("clap", {}).get("cosine_similarity") is not None]) if any(r.get("clap", {}).get("cosine_similarity") is not None for r in practical) else None,
        },
        "musicbench": {
            "items": len(external),
            "clap_mean": statistics.mean([r["clap"]["cosine_similarity"] for r in external if r.get("clap", {}).get("cosine_similarity") is not None]) if any(r.get("clap", {}).get("cosine_similarity") is not None for r in external) else None,
            "clap_median": statistics.median([r["clap"]["cosine_similarity"] for r in external if r.get("clap", {}).get("cosine_similarity") is not None]) if any(r.get("clap", {}).get("cosine_similarity") is not None for r in external) else None,
            "tempo_mae_bpm": statistics.mean(bpm_err) if bpm_err else None,
            "tempo_within_10_bpm": (sum(bool(x) for x in bpm_ok) / len(bpm_ok)) if bpm_ok else None,
            "key_accuracy_estimator": (sum(bool(x) for x in key_ok) / len(key_ok)) if key_ok else None,
        },
        "performance": {
            "generated_items": len(times),
            "generation_seconds_median": statistics.median(times) if times else None,
            "audio_seconds_generated": sum(float(x or 0) for x in audio_secs),
            "audio_seconds_per_generation_second": (sum(float(x or 0) for x in audio_secs) / sum(times)) if times and sum(times) > 0 else None,
            "total_measured_energy_wh": sum(energy) if energy else None,
        },
        "evaluation": {"clap_items": len(clap), "clap_model": "laion/clap-htsat-fused"},
    }


def run_music(*, repo_root: Path, run_dir: Path, music_config: Path, profile: str, is_completed, mark_completed, should_stop, set_current) -> dict[str, Any]:
    prep = _prepared(run_dir)
    configs = parse_music_models(music_config)
    tasks_doc = json.loads((repo_root / "benchmarks/music/fixtures/practical/tests.json").read_text(encoding="utf-8"))
    tasks = {x["id"]: x for x in tasks_doc["tests"]}
    mb = load_json(Path(prep["musicbench_manifest"]))
    melody = Path(prep["melody_fixture"]["path"])
    result: dict[str, Any] = {
        "profile": profile,
        "musicbench_revision": mb["revision"],
        "musicbench_count": mb["count"],
        "clap": {k: v for k, v in prep["clap"].items() if k != "local_path"},
        "configurations": [],
    }

    set_current({"benchmark": "music", "model": "laion/clap-htsat-fused", "mode": "evaluator", "test": "load", "repeat": 1, "repeats": 1})
    clap = ClapScorer(prep)
    result["clap"]["load_seconds"] = clap.load_seconds

    for cfg in configs:
        if should_stop():
            return {"stopped": True, **result}
        prepared = _prepared_model(prep, cfg)
        base = run_dir / "raw" / "music" / cfg["model"].replace("/", "__")
        practical_rows: list[dict[str, Any]] = []
        external_rows: list[dict[str, Any]] = []
        generator: MusicGenGenerator | None = None

        if not prepared or prepared.get("status") != "ready":
            reason = "Model ontbreekt uit music preflight of backend is niet ondersteund"
            for tid in PRACTICAL[profile]:
                key = f"music|practical|{cfg['model']}|{cfg['revision']}|{tid}"
                p = base / "unsupported" / "practical" / f"{tid}.json"
                if not is_completed(key):
                    rr = {"type": "music-practical", "status": "unsupported", "model": cfg["model"], "test": tid, "reason": reason, "pass": None}
                    _write_result(p, rr); mark_completed(key, rr)
                practical_rows.append(load_json(p))
            for sample in mb["samples"]:
                key = f"music|musicbench|{cfg['model']}|{cfg['revision']}|{sample['source_index']}"
                p = base / "unsupported" / "musicbench" / f"{sample['source_index']}.json"
                if not is_completed(key):
                    rr = {"type": "musicbench", "status": "unsupported", "model": cfg["model"], "source_index": sample["source_index"], "reason": reason, "pass": None}
                    _write_result(p, rr); mark_completed(key, rr)
                external_rows.append(load_json(p))
            result["configurations"].append({"model": cfg["model"], "status": "unsupported", "reason": reason})
            continue

        try:
            set_current({"benchmark": "music", "model": cfg["model"], "mode": "generation", "test": "load-model", "repeat": 1, "repeats": 1})
            generator = MusicGenGenerator(prepared, cfg)
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            for tid in PRACTICAL[profile]:
                key = f"music|practical|{cfg['model']}|{cfg['revision']}|{tid}"
                p = base / "unsupported" / "practical" / f"{tid}.json"
                if not is_completed(key):
                    rr = {"type": "music-practical", "status": "unsupported", "model": cfg["model"], "test": tid, "reason": reason, "pass": None}
                    _write_result(p, rr); mark_completed(key, rr)
                practical_rows.append(load_json(p))
            for sample in mb["samples"]:
                key = f"music|musicbench|{cfg['model']}|{cfg['revision']}|{sample['source_index']}"
                p = base / "unsupported" / "musicbench" / f"{sample['source_index']}.json"
                if not is_completed(key):
                    rr = {"type": "musicbench", "status": "unsupported", "model": cfg["model"], "source_index": sample["source_index"], "reason": reason, "pass": None}
                    _write_result(p, rr); mark_completed(key, rr)
                external_rows.append(load_json(p))
            result["configurations"].append({"model": cfg["model"], "status": "unsupported", "error": reason})
            continue

        try:
            # Practical prompts
            for tid in PRACTICAL[profile]:
                if should_stop():
                    return {"stopped": True, **result}
                task = tasks[tid]
                key = f"music|practical|{cfg['model']}|{cfg['revision']}|{tid}"
                path = base / "practical" / f"{tid}.json"
                if is_completed(key) and path.exists():
                    practical_rows.append(load_json(path)); continue
                set_current({"benchmark": "music", "model": cfg["model"], "mode": "generation", "test": tid, "repeat": 1, "repeats": 1})
                capability = task.get("capability", "text")
                if capability not in cfg["capabilities"]:
                    rr = {
                        "type": "music-practical", "test": tid, "title": task["title"], "model": cfg["model"],
                        "status": "unsupported", "required_capability": capability, "model_capabilities": sorted(cfg["capabilities"]),
                        "human_review": task.get("human_review", []), "pass": None, "checks_passed": None, "checks_total": None,
                    }
                    _write_result(path, rr); mark_completed(key, rr); practical_rows.append(rr); continue
                wav = base / "audio" / "practical" / f"{tid}.wav"
                gen = generator.generate(
                    prompt=task["prompt"], duration=float(task["duration_seconds"]), output=wav,
                    melody=melody if capability == "melody" else None, seed=SEED,
                )
                if gen.get("ok"):
                    feat = audio_features(wav)
                    clap_score = clap.score(task["prompt"], wav)
                    checks = _objective_checks(task, feat)
                    mel_sim = melody_chroma_similarity(melody, wav) if capability == "melody" else None
                    rr = {
                        "type": "music-practical", "test": tid, "title": task["title"], "model": cfg["model"],
                        "revision": cfg["revision"], "resolved_revision": generator.resolved_revision,
                        "backend": cfg["backend"], "prompt": task["prompt"], "seed": SEED, "generation": gen,
                        "audio_features": feat, "clap": clap_score, "objective_checks": checks,
                        "melody_chroma_similarity_proxy": mel_sim, "human_review": task.get("human_review", []),
                        "pass": all(x["passed"] for x in checks) if checks else None,
                        "checks_passed": sum(bool(x["passed"]) for x in checks) if checks else None,
                        "checks_total": len(checks) if checks else None, "completed_at": utc_now(),
                    }
                else:
                    rr = {
                        "type": "music-practical", "test": tid, "title": task["title"], "model": cfg["model"],
                        "prompt": task["prompt"], "generation": gen, "status": "error", "human_review": task.get("human_review", []),
                        "pass": None, "checks_passed": None, "checks_total": None, "completed_at": utc_now(),
                    }
                _write_result(path, rr); mark_completed(key, rr); practical_rows.append(rr)

            # External MusicBench prompt/control layer. 10 s fixed generation for comparable CLAP/tempo/key metrics.
            for i, sample in enumerate(mb["samples"]):
                if should_stop():
                    return {"stopped": True, **result}
                key = f"music|musicbench|{cfg['model']}|{cfg['revision']}|{sample['source_index']}"
                path = base / "external" / "musicbench" / f"{i:04d}.json"
                if is_completed(key) and path.exists():
                    external_rows.append(load_json(path)); continue
                set_current({"benchmark": "music", "model": cfg["model"], "mode": "generation", "test": f"MusicBench-{i+1}/{len(mb['samples'])}", "repeat": 1, "repeats": 1})
                if "text" not in cfg["capabilities"]:
                    rr = {"type": "musicbench", "status": "unsupported", "model": cfg["model"], "source_index": sample["source_index"], "reason": "text capability ontbreekt", "pass": None}
                    _write_result(path, rr); mark_completed(key, rr); external_rows.append(rr); continue
                wav = base / "audio" / "musicbench" / f"{i:04d}.wav"
                gen = generator.generate(prompt=sample["prompt"], duration=10.0, output=wav, melody=None, seed=SEED)
                if gen.get("ok"):
                    feat = audio_features(wav)
                    clap_score = clap.score(sample["prompt"], wav)
                    bpm_err = abs(float(feat["estimated_bpm"]) - float(sample["bpm"])) if feat.get("estimated_bpm") is not None and sample.get("bpm") is not None else None
                    key_match = _norm_key(feat.get("estimated_key")) == _norm_key(sample.get("key")) if feat.get("estimated_key") and sample.get("key") else None
                    control = {
                        "reference_bpm": sample.get("bpm"), "estimated_bpm": feat.get("estimated_bpm"),
                        "bpm_absolute_error": bpm_err, "bpm_within_10": bpm_err <= 10 if bpm_err is not None else None,
                        "reference_key": sample.get("key"), "estimated_key": feat.get("estimated_key"), "key_match": key_match,
                    }
                    rr = {
                        "type": "MusicBench-local-eval", "dataset": mb["source"], "dataset_revision": mb["revision"],
                        "source_index": sample["source_index"], "model": cfg["model"], "revision": cfg["revision"],
                        "resolved_revision": generator.resolved_revision, "prompt": sample["prompt"], "source_metadata": sample,
                        "seed": SEED, "generation": gen, "audio_features": feat, "clap": clap_score, "music_control": control,
                        "pass": None, "checks_passed": None, "checks_total": None, "completed_at": utc_now(),
                    }
                else:
                    rr = {"type": "MusicBench-local-eval", "dataset": mb["source"], "dataset_revision": mb["revision"], "source_index": sample["source_index"], "model": cfg["model"], "prompt": sample["prompt"], "generation": gen, "status": "error", "pass": None}
                _write_result(path, rr); mark_completed(key, rr); external_rows.append(rr)

            result["configurations"].append(_summary(cfg, generator, practical_rows, external_rows))
        finally:
            if generator is not None:
                generator.close()

    return {"stopped": should_stop(), **result}
