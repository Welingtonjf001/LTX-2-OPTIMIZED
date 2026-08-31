"""Run-folder convention for the screenplay pipeline.

Mirrors ``create_generation_folder``/``run_subdir`` in ``music_maker_ui_v2.py``: one
timestamped folder per run under ``outputs/screenplay/``, with a fixed set of stage
subfolders and a JSON manifest tracking which stages have completed -- the mechanism
``screenplay_to_video.py --resume-from`` uses to skip already-finished stages.
"""

from __future__ import annotations

import datetime
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "outputs" / "screenplay"

# Every stage of the pipeline gets its own subfolder, created up front so later
# stages never have to guess whether their output directory exists yet.
STAGE_DIRS = (
    "input",       # copy of the source screenplay
    "parse",       # scenes.json (structural) + scenes_enriched.json (LLM-enriched)
    "characters",  # cast.json (editable casting sheet)
    "storyboard",  # scene_XX.png
    "dialogue",    # scene_XX_line_YY.wav (dry TTS per line)
    "scenes",      # scene_XX[_line_YY].mp4 (LTX render output)
    "lipsync",     # scene_XX[_line_YY]_synced.mp4
    "intermediate",# scratch: concat lists, spatialized audio, etc.
    "final",       # movie.mp4
    "logs",        # per-run text log
)


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return slug or "screenplay"


def manifest_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "generation_manifest.json"


def read_manifest(run_dir: Path | str) -> dict:
    path = manifest_path(run_dir)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(run_dir: Path | str, manifest: dict) -> None:
    with open(manifest_path(run_dir), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def create_run(script_path: str, *, run_root: Path | None = None) -> Path:
    """Create ``outputs/screenplay/<timestamp>_<slug>/`` and snapshot the screenplay.

    Returns the new run directory. Mirrors ``create_generation_folder``'s
    collision-avoidance (append ``_1``, ``_2``, ... if the timestamped name is taken).
    """
    root = run_root or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    stem = Path(script_path).stem
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{stamp}_{slugify(stem)}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{slugify(stem)}_{suffix}"
        suffix += 1

    for name in STAGE_DIRS:
        (candidate / name).mkdir(parents=True, exist_ok=True)

    script_source = Path(script_path).resolve()
    shutil.copy2(script_source, candidate / "input" / script_source.name)

    write_manifest(candidate, {
        "created_at": datetime.datetime.now().isoformat(),
        "run_directory": str(candidate),
        "script_source": str(script_source),
        "stages_completed": [],
    })
    return candidate


def subdir(run_dir: Path | str, name: str) -> Path:
    """Return (creating if needed) one of the fixed stage subfolders.

    Always absolute: a relative run_dir crossing into a subprocess with a
    DIFFERENT cwd (e.g. LatentSync's own subprocess, cwd=LATENTSYNC_ROOT) silently
    resolves against the wrong directory -- this bit lipsync_scenes.py once already
    (see the fix in tensorxx_ge/lipsync.py). Resolving here closes it at the source
    for every stage script that goes through this helper.
    """
    path = (Path(run_dir) / name).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def mark_stage_complete(run_dir: Path | str, stage: str) -> None:
    manifest = read_manifest(run_dir)
    stages = manifest.setdefault("stages_completed", [])
    if stage not in stages:
        stages.append(stage)
    manifest["updated_at"] = datetime.datetime.now().isoformat()
    write_manifest(run_dir, manifest)


def stage_complete(run_dir: Path | str, stage: str) -> bool:
    return stage in read_manifest(run_dir).get("stages_completed", [])


def log_path(run_dir: Path | str) -> Path:
    return subdir(run_dir, "logs") / "pipeline.log"


def append_log(run_dir: Path | str, line: str) -> None:
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    text = f"[{timestamp}] {line}"
    print(text, flush=True)
    with open(log_path(run_dir), "a", encoding="utf-8") as handle:
        handle.write(text + "\n")
