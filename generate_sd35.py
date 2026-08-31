#!/usr/bin/env python3
"""Generate an image with SD 3.5 via ComfyUI API."""

import os, sys, json, time, urllib.parse, urllib.request, urllib.error

BASE = r"E:\Users\home\Documents\LTX-2-OPTIMIZED"
MODELS = os.path.join(BASE, "models")
COMFY_URL = "http://127.0.0.1:8188"

# ── 1. Download VAE if missing ────────────────────────────────────────
vae_path = os.path.join(MODELS, "ae.safetensors")
local_sd3_vae = os.path.join(MODELS, "Sd3.VAE.safetensors")
if not os.path.exists(vae_path) and os.path.exists(local_sd3_vae):
    vae_path = local_sd3_vae
if not os.path.exists(vae_path):
    print("Downloading SD 3.5 VAE (ae.safetensors) ...", file=sys.stderr)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="stabilityai/stable-diffusion-3.5-medium",
        filename="ae.safetensors",
        local_dir=MODELS,
    )

# ── 2. Verify all components ─────────────────────────────────────────
required = {
    "main": os.path.join(MODELS, "sd3.5_medium.safetensors"),
    "clip_l": os.path.join(MODELS, "clip_l.safetensors"),
    "t5xxl": os.path.join(MODELS, "t5xxl_fp8_e4m3fn.safetensors"),
    "vae": vae_path,
}
for name, p in required.items():
    if not os.path.exists(p):
        print(f"MISSING: {name} -> {p}", file=sys.stderr)
        sys.exit(1)
    print(f"  OK  {name}: {os.path.getsize(p)//1024//1024} MB")

# ── 3. Check ComfyUI server ──────────────────────────────────────────
def comfy_get(path):
    try:
        with urllib.request.urlopen(f"{COMFY_URL}{path}", timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"ComfyUI not reachable at {COMFY_URL}: {e}", file=sys.stderr)
        sys.exit(1)

sys_info = comfy_get("/system_stats")
print(f"\nComfyUI OK — GPU: {sys_info.get('gpu', {})}")

# ── 4. Build workflow for SD 3.5 Medium (unified checkpoint + separate encoders) ──
# SD 3.5 Medium uses a unified checkpoint with separate text encoders.
prompt_text = "A Korean female student sitting and waiting for the bus, school uniform, backpack, realistic photo, detailed face, natural lighting"
negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy"

workflow = {
    "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "sd3.5_medium.safetensors",
            "weight_dtype": "default",
        },
    },
    "2": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
            "model": ["1", 0],
            "shift": 3.0,
        },
    },
    "3": {
        "class_type": "CLIPTextEncodeSD3",
        "inputs": {
            "clip": ["10", 0],
            "clip_l": prompt_text,
            "clip_g": prompt_text,
            "t5xxl": prompt_text,
            "empty_padding": "none",
        },
    },
    "4": {
        "class_type": "CLIPTextEncodeSD3",
        "inputs": {
            "clip": ["10", 0],
            "clip_l": negative_prompt,
            "clip_g": negative_prompt,
            "t5xxl": negative_prompt,
            "empty_padding": "empty_prompt",
        },
    },
    "5": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": 768,
            "height": 512,
            "batch_size": 1,
        },
    },
    "6": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["2", 0],
            "positive": ["3", 0],
            "negative": ["4", 0],
            "latent_image": ["5", 0],
            "seed": 42,
            "steps": 28,
            "cfg": 4.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
    },
    "7": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "Sd3.VAE.safetensors",
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["6", 0],
            "vae": ["7", 0],
        },
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["8", 0],
            "filename_prefix": "sd35_output",
        },
    },
    "10": {
        "class_type": "TripleCLIPLoader",
        "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "clip_g.safetensors",
            "clip_name3": "t5xxl_fp8_e4m3fn.safetensors",
        },
    },
}

# ── 5. Submit workflow ───────────────────────────────────────────────
print("\nSubmitting workflow ...")
data = json.dumps({"prompt": workflow, "client_id": "sd35-gen"}).encode()
req = urllib.request.Request(
    f"{COMFY_URL}/prompt",
    data=data,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req) as r:
        result = json.loads(r.read())
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", errors="replace")
    print(f"ComfyUI rejected workflow: HTTP {e.code} {e.reason}", file=sys.stderr)
    print(body, file=sys.stderr)
    sys.exit(1)
prompt_id = result["prompt_id"]
print(f"Prompt ID: {prompt_id}")

# ── 6. Poll for completion ───────────────────────────────────────────
print("Waiting for generation to complete ...")
while True:
    time.sleep(2)
    history = comfy_get(f"/history/{prompt_id}")
    if prompt_id in history:
        status = history[prompt_id].get("status", {})
        print(f"Done! Status: {status.get('status_str', 'unknown')}")

        # Find output images
        outputs = history[prompt_id].get("outputs", {})
        for node_id, node_out in outputs.items():
            if "images" in node_out:
                for img in node_out["images"]:
                    out_path = os.path.join(BASE, f"sd35_output_{img['filename']}")
                    # Download image from ComfyUI
                    img_query = urllib.parse.urlencode({
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })
                    req2 = urllib.request.Request(f"{COMFY_URL}/view?{img_query}")
                    with urllib.request.urlopen(req2) as r2:
                        with open(out_path, "wb") as f:
                            f.write(r2.read())
                    print(f"Saved: {out_path}")
        break
