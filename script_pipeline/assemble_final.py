"""Stage [8]: concatenate every clip, in script order, into the final movie.

Same ffmpeg concat-demuxer pattern already established in music_maker_ui_v2/v3.py:
try a fast stream-copy concat first, fall back to a full re-encode if the clips'
codecs/parameters don't match closely enough for stream copy to work.

CLI: python -m script_pipeline.assemble_final --run-dir DIR [--output NAME]
Reads <run>/intermediate/mixed_clips.json (produced by mix_audio.py -- run that stage
even with no reverb/ambient requested, since it's what normalizes "final per-clip
video" into one field regardless of whether audio processing actually ran).
Writes <run>/final/<output>.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_mixed_clips(run_dir: Path) -> list[dict]:
    path = run_dir / "intermediate" / "mixed_clips.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run mix_audio first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _has_audio_stream(video_path: str, *, ffmpeg: str) -> bool:
    ffprobe = os.environ.get("LTX_FFPROBE", str(Path(ffmpeg).with_name("ffprobe.exe")))
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return bool(result.stdout.strip())


# Every clip fed to the concat demuxer must share ONE audio format.
CONCAT_AUDIO_RATE = "48000"
CONCAT_AUDIO_CHANNELS = "2"


def _normalize_audio_for_concat(video_path: str, work_dir: Path, *, ffmpeg: str, log) -> str:
    """Return a copy of this clip whose audio is AAC/48kHz/stereo, inventing a silent
    track when the clip has none. Video is stream-copied (only audio is re-encoded).

    MEASURED (2026-08-10), in two rounds -- the second because the first fix looked
    right and wasn't:
      1. The concat demuxer takes its output stream layout from the FIRST input. A
         film opening on a wordless action shot (video-only, no TTS audio to mux) came
         out with no audio track at all.
      2. Adding a silent track to only those clips was NOT enough: matching the
         *presence* of a stream doesn't matter, matching its *parameters* does. LTX
         writes dialogue audio as PCM at 16 kHz; the silence generated here was AAC at
         48 kHz. With -c copy the demuxer kept the first file's parameters and dropped
         the rest, so the film had a proper-looking AAC track measuring -91 dB --
         digital silence end to end. Verified with ffmpeg volumedetect, which is the
         check that catches this and a stream listing is not.
    Hence: normalise EVERY clip to identical audio parameters, not just the silent ones.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / (Path(video_path).stem + "_norm.mp4")
    if _has_audio_stream(video_path, ffmpeg=ffmpeg):
        command = [
            ffmpeg, "-y", "-v", "error", "-i", video_path,
            "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", CONCAT_AUDIO_RATE, "-ac", CONCAT_AUDIO_CHANNELS,
            str(out_path),
        ]
    else:
        command = [
            ffmpeg, "-y", "-v", "error", "-i", video_path,
            "-f", "lavfi", "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={CONCAT_AUDIO_RATE}",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", CONCAT_AUDIO_RATE, "-ac", CONCAT_AUDIO_CHANNELS,
            "-shortest", str(out_path),
        ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not out_path.exists():
        log(f"{Path(video_path).name}: falha ao normalizar audio ({result.stderr[-300:]}); usando original.")
        return video_path
    return str(out_path)


def concat_videos(video_paths: list[str], output_path: Path, *, work_dir: Path, log) -> bool:
    ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    work_dir.mkdir(parents=True, exist_ok=True)

    norm_dir = work_dir / "_audio_normalized"
    n_silent = sum(1 for p in video_paths if not _has_audio_stream(p, ffmpeg=ffmpeg))
    video_paths = [_normalize_audio_for_concat(p, norm_dir, ffmpeg=ffmpeg, log=log) for p in video_paths]
    log(f"assemble_final: audio normalizado para AAC {CONCAT_AUDIO_RATE}Hz estereo em "
        f"{len(video_paths)} clipe(s) ({n_silent} sem audio receberam silencio). "
        "Sem isso o concat descarta o audio dos clipes que divergem do primeiro.")

    concat_list = work_dir / "concat_list.txt"
    with open(concat_list, "w", encoding="ascii") as handle:
        for path in video_paths:
            safe_path = os.path.abspath(path).replace("\\", "/")
            handle.write(f"file '{safe_path}'\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", str(output_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0 or not output_path.exists():
        log("Concat direto (stream copy) falhou; reencodificando H.264/AAC.")
        result = subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "256k", str(output_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    if result.returncode != 0 or not output_path.exists():
        log(f"Concat falhou: {result.stderr[-2000:]}")
        return False
    return True


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="movie.mp4")
    args = parser.parse_args(argv)

    from script_pipeline import run_folder

    run_dir = Path(args.run_dir).resolve()
    clips = _load_mixed_clips(run_dir)
    final_dir = run_folder.subdir(run_dir, "final")
    log = lambda msg: run_folder.append_log(run_dir, msg)  # noqa: E731

    video_paths = [c["mixed_video_path"] for c in clips if c.get("mixed_video_path")]
    missing = [c["id"] for c in clips if not c.get("mixed_video_path")]
    if missing:
        log(f"assemble_final: {len(missing)} clipe(s) sem video final, nao incluidos: {missing}")
    if not video_paths:
        log("assemble_final: nenhum clipe disponivel para montar.")
        return 1

    output_path = final_dir / args.output
    log(f"assemble_final: concatenando {len(video_paths)} clipe(s) em ordem de roteiro -> {output_path}")
    ok = concat_videos(video_paths, output_path, work_dir=run_folder.subdir(run_dir, "intermediate"), log=log)

    if not ok:
        log("assemble_final: FALHOU.")
        return 1

    log(f"assemble_final: filme final pronto -> {output_path}")
    run_folder.mark_stage_complete(run_dir, "assemble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
