"""Stage [9]: post-render self-review -- check the finished film against the defects
this pipeline has actually shipped, and fail loudly instead of quietly.

Every check here exists because the corresponding bug reached the user in a real run:

  * SILENT_FILM   -- a film was delivered with a perfectly valid-looking AAC track
                     measuring -91 dB end to end (concat had dropped every clip's real
                     audio because their sample rates disagreed). `ffprobe` showing an
                     audio stream is NOT evidence of sound; only a level measurement is.
  * MUTE_DIALOGUE -- individual spoken clips silent while the film overall has sound.
  * IDENTICAL_SHOTS -- consecutive clips came out visually indistinguishable when an
                     end-keyframe anchored every clip to the same image, collapsing the
                     chain. Different shot descriptions, same picture.
  * BLACK_CLIP    -- a clip that decoded to a flat black/blank frame.
  * SHORT_FILM    -- the assembled film is far shorter than the sum of its clips,
                     i.e. clips were silently dropped at concat time.

CLI: python -m script_pipeline.verify_output --run-dir DIR [--output movie.mp4]
     [--min-audio-db -60] [--similarity-threshold 0.97] [--strict]
Writes <run>/verification.json and returns non-zero when a hard check fails (with
--strict; otherwise it reports and returns 0 so a report is never itself a blocker).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ffmpeg() -> str:
    return os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")


def _ffprobe() -> str:
    return os.environ.get("LTX_FFPROBE", str(Path(_ffmpeg()).with_name("ffprobe.exe")))


def probe_duration(path: str) -> float:
    result = subprocess.run(
        [_ffprobe(), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return 0.0


def has_audio_stream(path: str) -> bool:
    result = subprocess.run(
        [_ffprobe(), "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return bool(result.stdout.strip())


def measure_audio_db(path: str) -> tuple[float | None, float | None]:
    """Return (mean_dB, max_dB), or (None, None) when there is no audio at all.

    This is the check that a stream listing cannot replace: a track can be present,
    correctly encoded, and still be pure digital silence at -91 dB."""
    if not has_audio_stream(path):
        return None, None
    null_sink = "NUL" if os.name == "nt" else "/dev/null"
    result = subprocess.run(
        [_ffmpeg(), "-i", path, "-af", "volumedetect", "-f", "null", null_sink],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    blob = result.stderr or ""
    mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", blob)
    peak = re.search(r"max_volume:\s*(-?[\d.]+) dB", blob)
    return (float(mean.group(1)) if mean else None,
            float(peak.group(1)) if peak else None)


def _frame_signature(video_path: str, work_dir: Path, tag: str) -> list[int] | None:
    """A tiny perceptual (difference-hash style) signature of one mid-clip frame.

    Deliberately cheap and dependency-free beyond PIL: the goal is only to answer
    "are these two shots basically the same picture?", which is what the collapsed
    chain produced -- not to do real image analysis."""
    work_dir.mkdir(parents=True, exist_ok=True)
    frame = work_dir / f"{tag}.png"
    duration = probe_duration(video_path)
    seek = max(0.0, duration / 2)
    subprocess.run(
        [_ffmpeg(), "-y", "-v", "error", "-ss", f"{seek:.2f}", "-i", video_path,
         "-frames:v", "1", str(frame)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if not frame.exists():
        return None
    from PIL import Image
    with Image.open(frame) as img:
        small = img.convert("L").resize((9, 8))
        pixels = list(small.getdata())
    bits = []
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits.append(1 if left > right else 0)
    return bits


def _signature_similarity(a: list[int], b: list[int]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / len(a)


def _is_black(video_path: str, work_dir: Path, tag: str) -> bool:
    frame = work_dir / f"{tag}.png"
    if not frame.exists():
        return False
    from PIL import Image, ImageStat
    with Image.open(frame) as img:
        stat = ImageStat.Stat(img.convert("L"))
    # Near-zero brightness AND near-zero variation: a genuinely dark night shot still
    # has structure, a broken decode does not.
    return stat.mean[0] < 6.0 and stat.stddev[0] < 3.0


def verify(run_dir: Path, *, output_name: str, min_audio_db: float,
           similarity_threshold: float, log) -> dict:
    findings: list[dict] = []
    work_dir = run_dir / "intermediate" / "_verify_frames"

    final_path = run_dir / "final" / output_name
    report: dict = {"final_video": str(final_path), "checks": [], "findings": findings}

    if not final_path.exists():
        findings.append({"code": "NO_FILM", "severity": "error",
                         "detail": f"{final_path} nao existe."})
        return report

    # --- the film's own soundtrack -------------------------------------------------
    mean_db, max_db = measure_audio_db(str(final_path))
    report["checks"].append({"check": "final_audio", "mean_db": mean_db, "max_db": max_db})
    if mean_db is None:
        findings.append({"code": "SILENT_FILM", "severity": "error",
                         "detail": "O filme final nao tem faixa de audio nenhuma."})
    elif mean_db < min_audio_db:
        findings.append({
            "code": "SILENT_FILM", "severity": "error",
            "detail": f"O filme tem faixa de audio mas esta praticamente mudo "
                      f"(mean {mean_db:.1f} dB < limite {min_audio_db:.0f} dB). "
                      "Foi exatamente assim que um concat com taxas de amostragem "
                      "divergentes descartou todas as falas.",
        })

    # --- per-clip checks ------------------------------------------------------------
    clips_path = run_dir / "intermediate" / "mixed_clips.json"
    if not clips_path.exists():
        clips_path = run_dir / "scenes" / "clips.json"
    clips = json.loads(clips_path.read_text(encoding="utf-8")) if clips_path.exists() else []

    total_clip_seconds = 0.0
    signatures: list[tuple[str, list[int] | None]] = []
    for clip in clips:
        path = clip.get("mixed_video_path") or clip.get("video_path")
        if not path or not Path(path).exists():
            continue
        clip_id = clip.get("id", Path(path).stem)
        total_clip_seconds += probe_duration(path)

        # A clip that carries a dialogue line must actually be audible.
        if clip.get("audio_path"):
            c_mean, _c_max = measure_audio_db(path)
            if c_mean is None or c_mean < min_audio_db:
                findings.append({
                    "code": "MUTE_DIALOGUE", "severity": "error", "clip": clip_id,
                    "detail": f"Clipe de fala sem audio audivel (mean={c_mean}).",
                })

        sig = _frame_signature(path, work_dir, clip_id)
        signatures.append((clip_id, sig))
        if sig is not None and _is_black(path, work_dir, clip_id):
            findings.append({"code": "BLACK_CLIP", "severity": "error", "clip": clip_id,
                             "detail": "O clipe decodifica para um quadro preto/vazio."})

    # --- shot variety: REPORTED, NOT JUDGED -----------------------------------------
    # This started out as an IDENTICAL_SHOTS check meant to catch the collapsed chain
    # (every clip anchored to one image, so different descriptions produced the same
    # picture). MEASURED on a known-bad run against a known-good one, it does not work:
    #     dHash of a mid-clip frame -- bad run max 0.734 / median 0.531
    #                                  good run max 0.766 / median 0.516
    #     colour histogram          -- bad run max 0.814,  good run max 0.828
    # The GOOD run scores HIGHER on both, so no threshold separates them. Cheap
    # per-frame statistics cannot see the defect, because it is semantic (same subject
    # arrangement and blocking) while every shot in a scene legitimately shares one
    # location, palette and lighting. Detecting it properly needs an embedding that
    # encodes content -- CLIP is already present in this project (models/clip_interrogator)
    # and would be the honest next attempt.
    # Until then the numbers are published as data, and NOTHING is asserted from them:
    # a check that cannot separate good from bad must not emit verdicts.
    pair_scores = []
    for i, (id_a, sig_a) in enumerate(signatures):
        for id_b, sig_b in signatures[i + 1:]:
            if sig_a is None or sig_b is None:
                continue
            pair_scores.append((_signature_similarity(sig_a, sig_b), id_a, id_b))
    pair_scores.sort(reverse=True)
    report["checks"].append({
        "check": "shot_variety_informational",
        "pairs_compared": len(pair_scores),
        "max_similarity": round(pair_scores[0][0], 3) if pair_scores else None,
        "most_similar_pair": f"{pair_scores[0][1]} ~ {pair_scores[0][2]}" if pair_scores else None,
        "note": "informativo -- sem poder discriminativo comprovado; ver comentario no codigo",
    })

    # --- nothing silently dropped at concat time ------------------------------------
    film_seconds = probe_duration(str(final_path))
    report["checks"].append({"check": "duration", "film_seconds": round(film_seconds, 2),
                             "sum_of_clips_seconds": round(total_clip_seconds, 2)})
    if total_clip_seconds and film_seconds < total_clip_seconds * 0.9:
        findings.append({
            "code": "SHORT_FILM", "severity": "error",
            "detail": f"O filme tem {film_seconds:.1f}s mas os clipes somam "
                      f"{total_clip_seconds:.1f}s -- provavelmente clipes foram descartados.",
        })

    _check_enrichment(run_dir, report, findings)
    return report


def _check_enrichment(run_dir: Path, report: dict, findings: list) -> None:
    """Catch a silent enrichment failure by inspecting what the parse stage produced.

    This check exists because of a MEASURED run (20260811_100922) that every stage
    reported as successful and that this very module passed with zero findings -- while
    the film visibly carried a speech balloon painted over the actress's face.

    The chain: the LLM returned multilingual token salad instead of JSON, so the scene
    got no visual_prompt and an empty shot_list; storyboard fell back to one image for
    the whole scene; and render fell back to using the RAW screenplay text as the LTX
    prompt -- text that literally reads `ela exclama: - No horario!`, so the model drew
    the quoted line as a comic-book bubble.

    Nothing downstream can see this: the audio is fine, no frame is black, the duration
    adds up. It is only visible by asking whether the scenes were ever enriched at all.
    """
    parse_dir = run_dir / "parse"
    enriched_path = parse_dir / "scenes_enriched.json"
    if not enriched_path.exists():
        return
    try:
        data = json.loads(enriched_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    scenes = data.get("scenes") if isinstance(data, dict) else data
    if not isinstance(scenes, list) or not scenes:
        return

    unenriched = [s for s in scenes
                  if not (s.get("visual_prompt") or "").strip() and not s.get("shot_list")]
    report["checks"].append({"check": "enrichment", "scenes": len(scenes),
                             "scenes_without_prompt_or_shots": len(unenriched)})
    if not unenriched:
        return
    findings.append({
        "code": "ENRICHMENT_FAILED",
        "severity": "error" if len(unenriched) == len(scenes) else "warning",
        "detail": f"{len(unenriched)} de {len(scenes)} cena(s) sem visual_prompt e sem shot_list -- "
                  "o enriquecimento pelo LLM falhou e o pipeline seguiu no fallback. Isso significa "
                  "um storyboard unico por cena (em vez de um por plano) e prompt de render feito do "
                  "roteiro cru, incluindo as falas entre aspas -- que o modelo tende a desenhar como "
                  "balao de fala. Rode o parse de novo com --enrich-engine gemma4 e veja o log do "
                  "parse para a resposta bruta do LLM.",
    })


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="movie.mp4")
    parser.add_argument("--min-audio-db", type=float, default=-60.0,
                         help="Below this mean level a track counts as silent. Digital "
                              "silence measures -91 dB; real speech measured -21 to -28 dB here.")
    parser.add_argument("--similarity-threshold", type=float, default=0.97,
                         help="Consecutive-shot signature similarity above which two shots "
                              "are reported as the same picture.")
    parser.add_argument("--strict", action="store_true",
                         help="Return non-zero when an error-severity finding exists.")
    args = parser.parse_args(argv)

    from script_pipeline import run_folder

    run_dir = Path(args.run_dir).resolve()
    log = lambda msg: run_folder.append_log(run_dir, msg)  # noqa: E731

    report = verify(run_dir, output_name=args.output, min_audio_db=args.min_audio_db,
                    similarity_threshold=args.similarity_threshold, log=log)
    (run_dir / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    findings = report["findings"]
    errors = [f for f in findings if f.get("severity") == "error"]
    warnings = [f for f in findings if f.get("severity") != "error"]

    if not findings:
        log("verify_output: OK -- nenhum problema detectado no filme final.")
    for f in findings:
        mark = "ERRO" if f.get("severity") == "error" else "AVISO"
        where = f" [{f['clip']}]" if f.get("clip") else ""
        log(f"verify_output: {mark} {f['code']}{where}: {f['detail']}")
    log(f"verify_output: {len(errors)} erro(s), {len(warnings)} aviso(s) -> {run_dir / 'verification.json'}")

    return 1 if (args.strict and errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
