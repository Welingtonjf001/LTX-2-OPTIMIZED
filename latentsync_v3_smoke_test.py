"""Six-second end-to-end LatentSync smoke test for Music Video V3."""

from __future__ import annotations

import datetime
import os

import music_maker_ui_v3 as app


def main() -> int:
    project = os.path.abspath(os.path.dirname(__file__))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project, "outputs", "music_video_v3", f"{stamp}_latentsync_smoke")
    for folder in ("input", "audio", "scenes", "frames", "lipsync", "intermediate", "final", "logs"):
        os.makedirs(os.path.join(run_dir, folder), exist_ok=True)
    app.CURRENT_RUN_DIR = run_dir

    source_video = os.path.join(project, "v2_smoke", "latentsync_source_6s.mp4")
    source_audio = os.path.join(project, "v2_smoke", "latentsync_audio_6s.wav")
    output_video = os.path.join(run_dir, "lipsync", "latentsync_v3_smoke_6s.mp4")
    result = app.run_latentsync(source_video, source_audio, output_video)
    print(f"SMOKE_RUN_DIR={run_dir}", flush=True)
    print(f"SMOKE_OUTPUT={result}", flush=True)
    print(app.CURRENT_LOG[-5000:], flush=True)
    return 0 if result and os.path.exists(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
