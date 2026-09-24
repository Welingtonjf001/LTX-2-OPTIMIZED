"""Auditoria técnica e de sincronização do animatic.

Confere duração, streams, cobertura dos stills e, para cada fala, localiza no
áudio AAC do animatic a forma de onda do WAV original. Assim a checagem mede o
arquivo realmente entregue, não apenas os tempos planejados.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import correlate

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
SAMPLE_RATE = 8000


def _probe(path: Path) -> dict:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _decode(path: Path, *, start: float | None = None, duration: float | None = None) -> np.ndarray:
    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def expected_cues(plan: dict, dialogue: list[dict]) -> list[dict]:
    by_key = {(int(x["scene_index"]), int(x["line_index"])): x
              for x in dialogue if x.get("ok") and x.get("audio_path")}
    cues = []
    cursor = 0.0
    for shot in plan.get("shots", []):
        seconds = float(shot.get("seconds") or 0)
        key = (int(shot.get("scene") or 0), int(shot.get("line_index"))) \
              if shot.get("line_index") is not None else None
        line = by_key.get(key) if key else None
        if line:
            audio_seconds = float(line.get("duration_sec") or 0)
            audio_path = line["audio_path"]
            if shot.get("audio_window"):
                from script_pipeline.speech_split import slice_wav
                window = shot["audio_window"]
                audio_path = slice_wav(audio_path, window)
                audio_seconds = float(window[1]) - float(window[0])
            lead = max(0.0, (seconds - audio_seconds) / 2.0)
            cues.append({
                "scene": key[0], "line": key[1], "shot": int(shot.get("index", len(cues))),
                "audio_path": audio_path, "audio_duration": audio_seconds,
                "shot_start": cursor, "shot_duration": seconds,
                "expected_start": cursor + lead,
                "margin_before": lead, "margin_after": seconds - audio_seconds - lead,
            })
        cursor += seconds
    return cues


def _locate(rendered: np.ndarray, source: np.ndarray, expected: float,
            search_seconds: float = 0.75) -> tuple[float, float]:
    src = source.astype(np.float64)
    src -= src.mean() if len(src) else 0
    pad = int(search_seconds * SAMPLE_RATE)
    expected_sample = int(expected * SAMPLE_RATE)
    lo = max(0, expected_sample - pad)
    hi = min(len(rendered), expected_sample + len(src) + pad)
    segment = rendered[lo:hi].astype(np.float64)
    segment -= segment.mean() if len(segment) else 0
    if len(src) < 32 or len(segment) < len(src):
        return expected, 0.0
    scores = correlate(segment, src, mode="valid", method="fft")
    lag = int(np.argmax(scores))
    aligned = segment[lag:lag + len(src)]
    denom = float(np.linalg.norm(aligned) * np.linalg.norm(src))
    confidence = float(scores[lag] / denom) if denom else 0.0
    return (lo + lag) / SAMPLE_RATE, confidence


def audit(run_dir: str | Path, *, tolerance: float = 0.12) -> dict:
    run = Path(run_dir).resolve()
    plan = json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    dialogue = json.loads((run / "dialogue" / "lines.json").read_text(encoding="utf-8"))
    animatic = run / "shots" / "animatic.mp4"
    probe = _probe(animatic)
    streams = probe.get("streams") or []
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    actual_duration = float((probe.get("format") or {}).get("duration") or 0)
    expected_duration = sum(float(s.get("seconds") or 0) for s in plan.get("shots", []))
    missing_stills = [int(s.get("index", i)) for i, s in enumerate(plan.get("shots", []))
                      if not __import__("script_pipeline.render_shots", fromlist=["x"]).still_candidates(run / "shots" / "stills", i)]
    cues = expected_cues(plan, dialogue)
    rendered = _decode(animatic) if has_audio else np.array([], dtype=np.float32)
    cue_reports = []
    for cue in cues:
        source = _decode(Path(cue["audio_path"]))
        detected, confidence = _locate(rendered, source, cue["expected_start"])
        error = detected - cue["expected_start"]
        ok = abs(error) <= tolerance and confidence >= 0.55
        cue_reports.append({**cue, "detected_start": round(detected, 4),
                            "offset_error": round(error, 4),
                            "correlation": round(confidence, 4), "pass": ok})
    failures = []
    if not has_video:
        failures.append("video_stream_missing")
    if cues and not has_audio:
        failures.append("audio_stream_missing")
    if abs(actual_duration - expected_duration) > max(0.08, 2 / float(plan.get("fps") or 24)):
        failures.append("duration_mismatch")
    if missing_stills:
        failures.append("missing_stills")
    if any(not cue["pass"] for cue in cue_reports):
        failures.append("dialogue_sync")
    report = {
        "status": "ok" if not failures else "blocked", "failures": failures,
        "animatic": str(animatic), "expected_duration": expected_duration,
        "actual_duration": actual_duration, "duration_error": actual_duration - expected_duration,
        "video_stream": has_video, "audio_stream": has_audio,
        "expected_stills": len(plan.get("shots", [])), "missing_stills": missing_stills,
        "dialogue_cues": cue_reports, "sync_tolerance_seconds": tolerance,
    }
    out = run / "shots" / "animatic_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=0.12)
    args = parser.parse_args(argv)
    report = audit(args.run_dir, tolerance=args.tolerance)
    print(f"[animatic_audit] status={report['status']}; "
          f"duracao={report['actual_duration']:.3f}/{report['expected_duration']:.3f}s; "
          f"stills ausentes={len(report['missing_stills'])}; "
          f"falas aprovadas={sum(x['pass'] for x in report['dialogue_cues'])}/"
          f"{len(report['dialogue_cues'])}")
    print(f"[animatic_audit] relatorio: {Path(args.run_dir) / 'shots' / 'animatic_audit.json'}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
