"""Qwen3-TTS batch worker. Runs INSIDE Qwen3-TTS's own isolated venv (.venv), not the
main project venv -- invoked via subprocess by dialogue_tts.py.

v1 scope: only the CustomVoice checkpoint (preset timbres + free-text emotion/delivery
"instruct") is wired up, since that's the one downloaded for the xtts-vs-Qwen
comparison. VoiceDesign (voice from a text description) and Base (3s clone from a
reference clip) are separate checkpoints, documented as a v2 upgrade, not built here.

Loads the model once, then generates the whole batch (Qwen3TTSModel.generate_custom_voice
natively accepts parallel lists for text/language/speaker/instruct, so the batch is one
generate() call, not a Python loop -- faster than xtts_worker's per-item loop).

Usage: <qwen venv python> qwen_worker.py --jobs jobs.json --results results.json
         [--model-path PATH]
jobs.json: list of {"id": str, "text": str, "language": str (e.g. "Portuguese"/"English"),
                     "speaker": str (one of the CustomVoice preset names), "instruct": str|null,
                     "output_path": str}
results.json (written on exit): list of {"id", "ok", "output_path", "error", "seconds"}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_MODEL_PATH = str(Path(r"E:\Users\home\Documents\Qwen3-TTS\models\0.6B-CustomVoice"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    results = []
    if not jobs:
        Path(args.results).write_text("[]", encoding="utf-8")
        return 0

    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    t_load = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash-attn is not installed in this venv
    )
    print(f"[qwen_worker] model loaded in {time.time() - t_load:.1f}s", flush=True)
    print(f"[qwen_worker] supported speakers: {tts.get_supported_speakers()}", flush=True)

    texts = [job["text"] for job in jobs]
    languages = [job.get("language", "Portuguese") for job in jobs]
    speakers = [job["speaker"] for job in jobs]
    instructs = [job.get("instruct") or "" for job in jobs]

    t0 = time.time()
    try:
        wavs, sr = tts.generate_custom_voice(
            text=texts, language=languages, speaker=speakers, instruct=instructs,
            max_new_tokens=2048,
        )
        batch_seconds = time.time() - t0
        for job, wav in zip(jobs, wavs):
            output_path = Path(job["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), wav, sr)
            results.append({
                "id": job.get("id", "?"), "ok": True, "output_path": str(output_path),
                "error": None, "seconds": batch_seconds / len(jobs),
            })
            print(f"[qwen_worker] {job.get('id', '?')} ok -> {output_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        # Batch call failed as a whole (e.g. one bad speaker name) -- fall back to
        # one-by-one so a single bad job doesn't lose the whole batch's output.
        print(f"[qwen_worker] batch generation failed ({exc}); retrying one by one", flush=True)
        for job in jobs:
            t1 = time.time()
            try:
                wavs, sr = tts.generate_custom_voice(
                    text=job["text"], language=job.get("language", "Portuguese"),
                    speaker=job["speaker"], instruct=job.get("instruct") or "",
                    max_new_tokens=2048,
                )
                output_path = Path(job["output_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), wavs[0], sr)
                results.append({
                    "id": job.get("id", "?"), "ok": True, "output_path": str(output_path),
                    "error": None, "seconds": time.time() - t1,
                })
            except Exception as inner_exc:  # noqa: BLE001
                results.append({
                    "id": job.get("id", "?"), "ok": False, "output_path": None,
                    "error": str(inner_exc), "seconds": time.time() - t1,
                })
                print(f"[qwen_worker] {job.get('id', '?')} FAILED: {inner_exc}", flush=True)

    Path(args.results).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
