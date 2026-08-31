# -*- coding: utf-8 -*-
"""Queue the remaining 5 screenplay prompts through the validated LTX-2.5
single-stage T2V/I2V ComfyUI workflow, one at a time, waiting for each.
"""
import json
import sys
import time
import urllib.request
import uuid

from _screenplay_prompts import NEGATIVE, PROMPTS

SERVER = "http://127.0.0.1:8188"
BASE_API = "_api_25_t2v_single.json"  # freshly converted, no prompt patched in yet


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def build_api(entry):
    api = json.load(open(BASE_API, encoding="utf-8"))

    # Structural fixes (same as the validated western run).
    api["5004:5545"]["inputs"]["clip_name"] = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    api["5014:4990"]["inputs"]["scale_method"] = "lanczos"
    inp = api["5014:4990"]["inputs"]
    inp.pop("width", None)
    inp.pop("height", None)
    inp.pop("crop", None)
    inp["resize_type"] = "scale dimensions"
    inp["resize_type.width"] = 512
    inp["resize_type.height"] = 512
    inp["resize_type.crop"] = "center"
    api["2004"]["inputs"]["image"] = "4d84bf57_i2v_input.png"

    # Prompt-specific.
    api["5508"]["inputs"]["value"] = entry["text"]
    api["5509"]["inputs"]["value"] = NEGATIVE
    api["5511"]["inputs"]["value"] = 24
    api["5512"]["inputs"]["value"] = entry["seconds"]
    api["5014:5506"]["inputs"]["value"] = False
    api["5514:3059"]["inputs"]["width"] = 768
    api["5514:3059"]["inputs"]["height"] = 512
    api["4852"]["inputs"]["filename_prefix"] = entry["prefix"]
    # SaveVideo.format/codec became COMFY_DYNAMICCOMBO_V3 in the ComfyUI core
    # update pulled mid-session (was a plain STRING/COMBO when the western
    # prompt's base conversion was made, which is why that one didn't need
    # this). Same dotted-key pattern as ResizeImageMaskNode.resize_type.
    api["4852"]["inputs"]["format"] = "auto"
    api["4852"]["inputs"]["format.codec"] = "auto"
    return api


def free_cache():
    """Best-effort: clear ComfyUI's intermediate-node cache before each submission.
    Belt-and-suspenders alongside launching the server with --cache-none -- a
    stale cache across unrelated /prompt submissions (different prompt/duration/
    seed each time) produced a reproducible shape-mismatch crash in the sampler
    (see MEMORIAL.md, LTX-2.5 section)."""
    try:
        post_json(f"{SERVER}/free", {"unload_models": False, "free_memory": True})
    except Exception as e:
        print(f"[free] non-fatal: {e}", file=sys.stderr, flush=True)


def run_one(name, entry, timeout=5400):
    api = build_api(entry)
    client_id = str(uuid.uuid4())
    free_cache()
    print(f"[{name}] submitting ({entry['seconds']}s)...", flush=True)
    try:
        res = post_json(f"{SERVER}/prompt", {"prompt": api, "client_id": client_id})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"[{name}] REJECTED: {body[:2000]}", file=sys.stderr, flush=True)
        return False
    pid = res.get("prompt_id")
    print(f"[{name}] prompt_id={pid}", flush=True)

    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = get_json(f"{SERVER}/history/{pid}")
        if pid in hist:
            entry_h = hist[pid]
            status = entry_h.get("status", {})
            smsg = status.get("status_str")
            elapsed = time.time() - t0
            if status.get("completed") or smsg == "success":
                outs = []
                for _nid, out in (entry_h.get("outputs") or {}).items():
                    for _key, items in out.items():
                        for it in items if isinstance(items, list) else []:
                            if isinstance(it, dict) and it.get("filename"):
                                outs.append(it["filename"])
                print(f"[{name}] DONE in {elapsed:.1f}s -> {outs}", flush=True)
                return True
            if smsg == "error":
                print(f"[{name}] FAILED in {elapsed:.1f}s: {json.dumps(status.get('messages', []))[:2000]}", flush=True)
                return False
        time.sleep(5)
    print(f"[{name}] TIMEOUT after {timeout}s", flush=True)
    return False


def main():
    names = sys.argv[1:] or list(PROMPTS.keys())
    results = {}
    for name in names:
        results[name] = run_one(name, PROMPTS[name])
    print("SUMMARY:", results, flush=True)


if __name__ == "__main__":
    main()
