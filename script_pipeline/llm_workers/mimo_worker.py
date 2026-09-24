"""MiMo-V2.6-Distill-Qwen-9B batch worker. Runs INSIDE the model's own isolated
venv (E:/Users/home/Documents/MiMo-V2.6-Distill-Qwen-9B/runtime -- transformers
5.17, needed for the qwen3_5 architecture; the main project venv doesn't have
it). Invoked via subprocess by parse_screenplay.py, mirroring gemma4_worker.py's
job-file/result-file pattern exactly.

Evaluated 2026-09-24 as an enrichment engine (see MEMORIAL): output quality is
sound (correct JSON schema, no hallucinated characters/locations, valid on the
real _apply_shot_list contract) but two measured tradeoffs versus the default
qwen2.5:32b-instruct-q4_K_M (via Ollama): ~55-65s per scene against ~41s, and a
tendency to over-segment the shot_list (12 shots against 9 for the identical
scene text) -- more rendered planos per scene for the same content. Kept
available as an opt-in choice, not promoted to any default.

Usage: <mimo runtime python> mimo_worker.py --jobs jobs.json --results results.json
         [--model-path PATH]
jobs.json: list of {"id": str, "system_prompt": str, "user_prompt": str,
                     "max_new_tokens": int}
results.json (written on exit): list of {"id", "ok", "raw_text", "error"}
"""

from __future__ import annotations

import argparse
import json
import time

DEFAULT_MODEL_PATH = r"E:\Users\home\Documents\MiMo-V2.6-Distill-Qwen-9B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    from pathlib import Path

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    results = []
    if not jobs:
        Path(args.results).write_text("[]", encoding="utf-8")
        return 0

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    # MEASURED 2026-09-24 in this venv: torch cuda:0 is the RTX 4070 (12.9GB),
    # cuda:1 is the RTX 3090 (25.8GB) -- same inverted-index convention already
    # documented in CLAUDE.md for this machine's other scripts. Loading the
    # whole ~18GB bf16 model onto cuda:1 alone was measured NOT slower than
    # letting device_map="auto" split it across both cards (in fact slightly
    # faster: no cross-GPU relay per layer), so pin to the 3090 explicitly
    # instead of trusting "auto" to pick it.
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": 1},
        local_files_only=True, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[mimo_worker] model loaded in {time.time() - t_load:.1f}s", flush=True)

    for job in jobs:
        t0 = time.time()
        try:
            messages = [
                {"role": "system", "content": job["system_prompt"]},
                {"role": "user", "content": job["user_prompt"]},
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            )
            inputs = inputs.to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                # Greedy, like the other two enrichment engines in this project
                # (Gemma3/Gemma4 both settled on do_sample=False for this task
                # after measuring sampling-related instability elsewhere) --
                # this is a structured-JSON extraction task, not creative
                # writing, so determinism is preferred over sampling variety.
                output = model.generate(
                    **inputs, max_new_tokens=job.get("max_new_tokens", 1200),
                    do_sample=False,
                )
            generated = output[0][input_len:]
            raw_text = processor.decode(generated, skip_special_tokens=True).strip()
            results.append({"id": job.get("id", "?"), "ok": True, "raw_text": raw_text, "error": None})
            print(f"[mimo_worker] {job.get('id', '?')} ok in {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"id": job.get("id", "?"), "ok": False, "raw_text": None, "error": str(exc)})
            print(f"[mimo_worker] {job.get('id', '?')} FAILED: {exc}", flush=True)

    Path(args.results).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
