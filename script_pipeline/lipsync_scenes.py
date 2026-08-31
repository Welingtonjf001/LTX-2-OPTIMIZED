"""Stage [6]: lip-sync each rendered dialogue clip to its own dry TTS line, via the
project's already-existing tensorxx_ge/lipsync.py (LatentSync 1.6 first, Wav2Lip
fallback -- unmodified, no fork needed: it's already generic to "one face, one dry
audio track per call", which is exactly what a per-line clip is).

Action-only clips (no dialogue, no audio) pass through untouched -- there is nothing
to sync.

CLI: python -m script_pipeline.lipsync_scenes --run-dir DIR [--engine auto]
Reads <run>/scenes/clips.json, writes <run>/lipsync/scene_XX[_line_YY]_synced.mp4 and
<run>/lipsync/synced_clips.json (id -> final video path, consumed by assemble_final.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_clips(run_dir: Path) -> list[dict]:
    clips_path = run_dir / "scenes" / "clips.json"
    if not clips_path.exists():
        raise FileNotFoundError(f"{clips_path} not found -- run render_scenes first.")
    return json.loads(clips_path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--engine", default="auto", choices=["auto", "latentsync", "wav2lip"])
    args = parser.parse_args(argv)

    from script_pipeline import run_folder
    from tensorxx_ge.lipsync import apply_lipsync, lipsync_status

    run_dir = Path(args.run_dir).resolve()
    clips = _load_clips(run_dir)
    lipsync_dir = run_folder.subdir(run_dir, "lipsync")

    log = lambda msg: run_folder.append_log(run_dir, msg)  # noqa: E731
    status = lipsync_status()
    log(f"lipsync_scenes: LatentSync {'disponivel' if status.latentsync_available else 'indisponivel'}, "
        f"Wav2Lip {'disponivel' if status.wav2lip_available else 'indisponivel'}.")

    synced_manifest = []
    failures = 0
    for clip in clips:
        if not clip.get("ok") or not clip.get("video_path"):
            log(f"{clip['id']}: sem video renderizado; pulando lip-sync.")
            synced_manifest.append({**clip, "final_video_path": None, "lipsync_applied": False})
            failures += 1
            continue

        if not clip.get("audio_path"):
            # Action-only clip: nothing to sync, pass through as-is.
            synced_manifest.append({**clip, "final_video_path": clip["video_path"], "lipsync_applied": False})
            log(f"{clip['id']}: sem fala (cena de acao); mantendo clipe original.")
            continue

        if not status.available:
            log(f"{clip['id']}: nenhum motor de lip-sync disponivel; mantendo clipe original.")
            synced_manifest.append({**clip, "final_video_path": clip["video_path"], "lipsync_applied": False})
            continue

        output_path = lipsync_dir / f"{clip['id']}_synced.mp4"
        result = apply_lipsync(
            clip["video_path"], clip["audio_path"], str(output_path),
            work_dir=str(lipsync_dir), engine=args.engine, log=log,
        )
        if result is not None:
            synced_manifest.append({**clip, "final_video_path": str(result), "lipsync_applied": True})
            log(f"{clip['id']}: lip-sync ok -> {result}")
        else:
            log(f"{clip['id']}: lip-sync falhou; mantendo clipe original sem sincronia labial.")
            synced_manifest.append({**clip, "final_video_path": clip["video_path"], "lipsync_applied": False})

    (lipsync_dir / "synced_clips.json").write_text(
        json.dumps(synced_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    ok_count = sum(1 for c in synced_manifest if c.get("final_video_path"))
    log(f"lipsync_scenes: {ok_count}/{len(clips)} clipe(s) com video final disponivel "
        f"({sum(1 for c in synced_manifest if c.get('lipsync_applied'))} com lip-sync aplicado).")

    if ok_count == len(clips):
        run_folder.mark_stage_complete(run_dir, "lipsync")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
