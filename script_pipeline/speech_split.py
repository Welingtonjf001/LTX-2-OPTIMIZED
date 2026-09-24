"""Cobertura de fala longa: divide uma fala em planos de ate N segundos.

Por que existe (avaliacao do Voo 702, 2026-09-18): falas de 12-18 s num unico
clipe generativo fazem o falante sair de quadro (o LTX "esquece" o sujeito) e,
sem rosto, o LatentSync falha com "Face not detected" -- o lipsync some do filme
inteiro em silencio. Planos de fala curtos mantem o rosto no quadro e o lipsync
viavel.

Todos os segmentos mantem o MESMO falante em close (o lipsync exige rosto). O
audio de cada segmento e um recorte do WAV da fala, cortado num silencio quando
existe um dentro do orcamento (nunca no meio de uma silaba), e viaja com o plano
em `audio_window` = [inicio, fim] em segundos DENTRO do wav da fala.

Consumidores usam `dialogue_entry(shot, dialogue)` no lugar de
`dialogue.get((cena, fala))`: ele devolve o recorte quando o plano tem janela.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

MAX_SPEECH_SECONDS = 6.0
MIN_SEGMENT_SECONDS = 1.6


def _ffmpeg() -> str:
    return os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")


def detect_silence_centers(wav: str, *, noise_db: float = -35.0, min_dur: float = 0.12) -> list[float]:
    """Centros (s) dos silencios internos do wav, via `silencedetect` do ffmpeg."""
    r = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-nostats", "-i", str(wav),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", r.stderr)]
    return [(a + b) / 2.0 for a, b in zip(starts, ends)]


def choose_boundaries(total: float, max_seconds: float, gap_centers: list[float],
                      min_segment: float = MIN_SEGMENT_SECONDS) -> list[float]:
    """Fronteiras [0, ..., total] com segmentos equilibrados e <= max_seconds.

    Comeca pelos cortes ideais (total/n) e encaixa cada um no silencio mais
    proximo dentro de +-35% do tamanho-alvo; se o encaixe estourar o teto ou
    deixar um segmento curto demais, o corte ideal (aritmetico) prevalece."""
    if total <= max_seconds:
        return [0.0, total]
    n = max(2, math.ceil(total / max_seconds))
    target = total / n
    bounds = [0.0]
    for k in range(1, n):
        ideal = k * target
        near = [c for c in gap_centers if abs(c - ideal) <= 0.35 * target and c > bounds[-1]]
        bounds.append(min(near, key=lambda c: abs(c - ideal)) if near else ideal)
    bounds.append(total)
    segs = [b - a for a, b in zip(bounds, bounds[1:])]
    if any(s > max_seconds + 1e-6 or s < min_segment for s in segs):
        bounds = [k * target for k in range(n)] + [total]
    return [round(b, 3) for b in bounds]


def split_long_speech(plan: dict, durations: dict, audio_paths: dict, *, max_seconds: float,
                      folga: float, frames_fn, fps: float, gap_fn=detect_silence_centers) -> dict:
    """Devolve o plano com as falas longas divididas em segmentos.

    `durations`/`audio_paths`: (cena, fala) -> segundos / caminho do wav. Fala
    sem duracao real (so estimativa) nao e dividida: os limites dependem do
    audio de verdade."""
    if not max_seconds or max_seconds <= 0:
        return plan
    novos, dividida = [], 0
    for shot in plan.get("shots", []):
        key = (shot.get("scene"), shot.get("line_index"))
        total = durations.get(key) if shot.get("line_index") is not None else None
        if not total or total <= max_seconds or shot.get("audio_window"):
            novos.append(shot)
            continue
        wav = audio_paths.get(key)
        centers = gap_fn(wav) if wav and Path(wav).exists() else []
        bounds = choose_boundaries(float(total), max_seconds, centers)
        n = len(bounds) - 1
        dividida += 1
        for k in range(n):
            seg = dict(shot)
            t0, t1 = bounds[k], bounds[k + 1]
            seg["audio_window"] = [t0, t1]
            seg["segment"] = k
            seg["segments"] = n
            seg["seconds"] = round((t1 - t0) + folga, 2)
            seg["frames"] = frames_fn(seg["seconds"], fps)
            novos.append(seg)
    plan = dict(plan)
    plan["shots"] = novos
    contador: dict = {}
    for i, shot in enumerate(novos):
        if "index" in shot:
            shot["index"] = i
        contador[shot.get("scene")] = contador.get(shot.get("scene"), -1) + 1
        shot["position"] = contador[shot.get("scene")]
    plan["total_shots"] = len(novos)
    plan["total_seconds"] = round(sum(float(s.get("seconds") or 0) for s in novos), 1)
    plan["speech_split"] = {"max_seconds": max_seconds, "falas_divididas": dividida}
    return plan


def load_audio_paths(lines_json) -> dict:
    if not lines_json or not Path(lines_json).exists():
        return {}
    out = {}
    for e in json.loads(Path(lines_json).read_text(encoding="utf-8")):
        if e.get("ok") and e.get("audio_path"):
            out[(e["scene_index"], e["line_index"])] = e["audio_path"]
    return out


def slice_wav(wav: str, window, *, fade: float = 0.01) -> str:
    """Recorte deterministico (cacheado) do wav em `window`; devolve o caminho."""
    t0, t1 = float(window[0]), float(window[1])
    src = Path(wav)
    out_dir = src.parent / "_slices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_{t0:.2f}_{t1:.2f}.wav"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return str(out)
    dur = t1 - t0
    af = f"afade=t=in:d={fade},afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade}"
    r = subprocess.run(
        [_ffmpeg(), "-y", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
         "-af", af, "-c:a", "pcm_s16le", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"falha ao recortar {src.name} [{t0}-{t1}]: {r.stderr[-300:]}")
    return str(out)


def dialogue_entry(shot: dict, dialogue: dict):
    """(wav, duracao) da fala DESTE plano, ou None. Recorta quando ha janela."""
    entrada = dialogue.get((shot.get("scene"), shot.get("line_index")))
    window = shot.get("audio_window")
    if not entrada or not entrada[0] or not window:
        return entrada
    if not Path(entrada[0]).exists():
        return entrada
    return (slice_wav(entrada[0], window), float(window[1]) - float(window[0]))
