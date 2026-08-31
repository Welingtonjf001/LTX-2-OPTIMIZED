"""Gemma 4 E2B batch worker. Runs INSIDE gemma4_env (isolated venv with transformers
5.14.1 -- Gemma3's Gemma3ForConditionalGeneration import needs transformers 4.x, and
Gemma 4's config.json requires transformers_version>=5.5.0.dev0; the two model
families can't share the main project venv without an upgrade that risks breaking
every other transformers-dependent path in this project (LTX's own hidden-state
encoder, xtts, Qwen3-TTS, CLIP interrogator). Invoked via subprocess by
parse_screenplay.py, mirroring tts_workers/qwen_worker.py's job-file/result-file
pattern exactly.

E2B is a single 10.2GB bf16 file -- unlike Gemma3 12B IT, no 8-bit quantization is
needed to fit the RTX 3090's 24GB, which also sidesteps the whole class of
quantization-related sampling instability measured with Gemma3 8-bit this session
(device-side CUDA asserts under top_k/top_p sampling). Loaded in native bf16.

Usage: <gemma4_env python> gemma4_worker.py --jobs jobs.json --results results.json
         [--model-path PATH]
jobs.json: list of {"id": str, "system_prompt": str, "user_prompt": str,
                     "max_new_tokens": int}
results.json (written on exit): list of {"id", "ok", "raw_text", "error"}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent.parent / "models" / "gemma4-e2b")


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
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    # GPU 1 (torch numbering) is the RTX 3090 (24GiB) in this installation -- same
    # convention already used for Gemma3 in parse_screenplay.py._load_gemma().
    torch.cuda.set_device(1)
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": 1},
    )
    model.eval()
    print(f"[gemma4_worker] model loaded in {time.time() - t_load:.1f}s", flush=True)

    for job in jobs:
        t0 = time.time()
        try:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": job["system_prompt"]}]},
                {"role": "user", "content": [{"type": "text", "text": job["user_prompt"]}]},
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                # Model's own generation_config.json recommends do_sample=true,
                # temperature=1.0, top_p=0.95, top_k=64 -- same params Gemma3's own
                # config recommends. Unlike the 8-bit Gemma3 setup (see
                # parse_screenplay.py's _ask_gemma docstring), this model is native
                # bf16 (no quantization), so the sampling-triggered CUDA device-side
                # assert measured there is not expected here -- if it reproduces
                # anyway, that would newly implicate sampling itself rather than
                # quantization specifically, worth knowing either way.
                output = model.generate(
                    **inputs, max_new_tokens=job.get("max_new_tokens", 400),
                    do_sample=True, temperature=1.0, top_p=0.95, top_k=64,
                )
            generated = output[0][input_len:]
            raw_text = processor.decode(generated, skip_special_tokens=True).strip()
            results.append({"id": job.get("id", "?"), "ok": True, "raw_text": raw_text, "error": None})
            print(f"[gemma4_worker] {job.get('id', '?')} ok in {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"id": job.get("id", "?"), "ok": False, "raw_text": None, "error": str(exc)})
            print(f"[gemma4_worker] {job.get('id', '?')} FAILED: {exc}", flush=True)

    Path(args.results).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
