"""ComfyUI backend for testing a LoRA on the LTX 2.3 route -- built for the
"Multi-Ref Character Storyboard V2" LoRA (models/loras/), but generic to any
LoraLoaderModelOnly-compatible LoRA on the 2.3 checkpoint.

Why a hand-built flat graph instead of importing the community workflow this
LoRA shipped with (a RunningHub post): that workflow uses ComfyUI subgraphs,
and `comfy_workflow_tool.convert()` has a KNOWN bug misreading widget values
inside subgraphs (see CLAUDE.md -- it already silently broke `batch_size` and
`LTXVImgToVideoInplace.strength` once). Hand-writing a flat API-format graph
sidesteps that converter entirely. Every node/param below was confirmed live
against this ComfyUI's own `/object_info` (not guessed from docs), including
the LoRA's own metadata (`ai-toolkit`, `base_model=ltx2`, rank-32 attention
LoRA -- a plain style/character LoRA, not an IC-LoRA needing special
conditioning wiring).

Reuses ltx25_backend's server lifecycle (ensure_server/submit_and_wait/
stage_input_image) instead of duplicating it: this project already enforces
"one shared ComfyUI instance" (CLAUDE.md -- running two competes for the same
VRAM), and ltx25_backend already owns the boot lock for port 8188.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
import urllib.error
import urllib.request

import ltx25_backend as backend

ROOT = backend.ROOT
COMFY_SERVER = backend.COMFY_SERVER
GEMMA_ROOT_DIR = os.path.join(ROOT, "models", "gemma3")

CHECKPOINT = "ltx-2.3-22b-distilled-fp8.safetensors"
VAE_PATH = "vae\\ltx-2.3-22b-dev_video_vae.safetensors"
LORA_DIR = os.path.join(ROOT, "models", "loras")

DEFAULT_NEGATIVE = (
    "low quality, worst quality, deformed, distorted, disfigured, motion smear, "
    "motion artifacts, fused fingers, bad anatomy, weird hand, ugly"
)

SAVE_NODE_ID = "13"


def list_loras() -> list[str]:
    if not os.path.isdir(LORA_DIR):
        return []
    return sorted(f for f in os.listdir(LORA_DIR) if f.endswith(".safetensors"))


def _encode_prompts_native(
    *, prompt: str, negative: str, pos_path: str, neg_path: str, log_cb=None,
) -> None:
    """Encode text with the NATIVE Gemma3 route (ltx_pipelines/model_ledger.py,
    the code path already proven to work, ~146s) instead of ComfyUI's own
    LTXVGemmaCLIPModelLoader.

    Why: investigated ComfyUI's own Gemma3 loading in depth (see
    memory/project_lora_storyboard_test_ui.md) -- root cause is
    `comfy.memory_management.aimdo_enabled` always forcing the CLIP to
    initialize on CPU (ignoring `--disable-dynamic-vram`, which is vestigial
    in this ComfyUI version), and the `--highvram` workaround that bypasses
    it lands at 24.1 of 24.5GB used -- fits, but hangs under load with no
    headroom. Reusing the native encoder sidesteps ComfyUI's Gemma3 path
    entirely; only the LoRA+video stage still goes through ComfyUI (that part
    was never broken). Runs as its own subprocess so CUDA memory is actually
    returned to the OS when it exits, same convention as every other
    GPU-heavy stage in this project (see CLAUDE.md, "Liberação de memória")."""
    def log(msg):
        backend._log(msg, log_cb)

    checkpoint_path = os.path.join(ROOT, "models", CHECKPOINT)
    upsampler_path = os.path.join(ROOT, "models", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PYTHONUNBUFFERED="1")

    # TWO separate subprocess calls, not one call encoding both prompts --
    # encoding [prompt, negative] together in a single encode_prompts() call
    # threw a `torch.OutOfMemoryError` reporting a physically impossible
    # "60.73 GiB allocated" on this 24GB card, reproduced 4x identically
    # (including with quantization/spatial_upsampler matched to production).
    # Production's own `music_to_video.py` only ever encodes ONE prompt per
    # call (`(context_p,) = encode_prompts([prompt], ...)` -- no negative);
    # matching that pattern here fixed it. See lora_storyboard_encode.py's
    # docstring for the full story.
    for label, text, out_path in (("positivo", prompt, pos_path), ("negativo", negative or DEFAULT_NEGATIVE, neg_path)):
        log(f"[lora_storyboard] Codificando prompt {label} pela rota nativa (Gemma3, ~100-150s a frio)...")
        cmd = [
            sys.executable, "-u", os.path.join(ROOT, "lora_storyboard_encode.py"),
            "--checkpoint", checkpoint_path,
            "--gemma-root", GEMMA_ROOT_DIR,
            "--spatial-upsampler-path", upsampler_path,
            "--prompt", text,
            "--out", out_path,
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env, timeout=600)
        output = (proc.stdout or "") + (proc.stderr or "")
        for line in output.splitlines():
            log(f"[lora_storyboard] {line}")
        if proc.returncode != 0:
            raise RuntimeError(f"Codificação do prompt {label} (rota nativa) falhou (código {proc.returncode}).")


def build_stage2_video(
    *,
    lora_name: str,
    lora_strength: float,
    pos_filename: str,
    neg_filename: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    steps: int,
    cfg: float,
    seed: int,
    image_filename: str | None,
    image_strength: float,
) -> dict:
    """Stage 2: load the saved conditioning + the 2.3 checkpoint + LoRA, run
    the actual video sampling. By the time this submits, stage 1 already
    finished and its models are the ones ComfyUI's LRU cache will evict first
    if this stage needs the room."""
    g: dict = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": CHECKPOINT, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": lora_name, "strength_model": lora_strength},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_PATH}},
        "21a": {
            "class_type": "LTXVLoadConditioning",
            "inputs": {"file_name": f"{pos_filename}.safetensors", "device": "gpu"},
        },
        "21b": {
            "class_type": "LTXVLoadConditioning",
            "inputs": {"file_name": f"{neg_filename}.safetensors", "device": "gpu"},
        },
    }

    if image_filename:
        g["7a"] = {"class_type": "LoadImage", "inputs": {"image": image_filename}}
        g["7b"] = {
            "class_type": "LTXVImgToVideo",
            "inputs": {
                "positive": ["21a", 0], "negative": ["21b", 0],
                "vae": ["4", 0], "image": ["7a", 0],
                "width": width, "height": height, "length": num_frames,
                "batch_size": 1, "strength": image_strength,
            },
        }
        positive_ref, negative_ref, latent_ref = ["7b", 0], ["7b", 1], ["7b", 2]
    else:
        g["7c"] = {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": width, "height": height, "length": num_frames, "batch_size": 1},
        }
        positive_ref, negative_ref, latent_ref = ["21a", 0], ["21b", 0], ["7c", 0]

    g["8"] = {
        "class_type": "LTXVConditioning",
        "inputs": {"positive": positive_ref, "negative": negative_ref, "frame_rate": float(frame_rate)},
    }
    g["9"] = {
        "class_type": "LTXVScheduler",
        "inputs": {
            "steps": steps, "max_shift": 2.05, "base_shift": 0.95,
            "stretch": True, "terminal": 0.1, "latent": latent_ref,
        },
    }
    g["10"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["11"] = {
        "class_type": "SamplerCustom",
        "inputs": {
            "model": ["2", 0], "add_noise": True, "noise_seed": int(seed), "cfg": float(cfg),
            "positive": ["8", 0], "negative": ["8", 1],
            "sampler": ["10", 0], "sigmas": ["9", 0], "latent_image": latent_ref,
        },
    }
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}}
    g[SAVE_NODE_ID] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["12", 0], "frame_rate": float(frame_rate), "loop_count": 0,
            # No subfolder in the prefix: ltx25_backend.submit_and_wait()'s file
            # list only returns bare filenames (no "subfolder" field from
            # ComfyUI's history), so a prefix like "dir/clip" produces a file
            # this script then can't find under ComfyUI/output/ directly
            # (confirmed live -- FileNotFoundError copying "dir/clip_00001.mp4"
            # from a path that omitted "dir/"). Flat prefix avoids that.
            "filename_prefix": "lora_storyboard_test_clip", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True,
        },
    }
    return g


def interrupt() -> str:
    """Cancel whatever ComfyUI is currently executing, and drop anything still
    queued. Does NOT kill the ComfyUI process itself -- it's the shared
    server other UIs in this project also depend on."""
    ok_parts = []
    try:
        req = urllib.request.Request(f"{COMFY_SERVER}/interrupt", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=10)
        ok_parts.append("execução atual interrompida")
    except Exception as exc:
        ok_parts.append(f"/interrupt falhou: {exc}")
    try:
        backend._post_json(f"{COMFY_SERVER}/queue", {"clear": True})
        ok_parts.append("fila limpa")
    except Exception as exc:
        ok_parts.append(f"limpar fila falhou: {exc}")
    return "; ".join(ok_parts)


def generate(
    *,
    prompt: str,
    output_dir: str,
    lora_name: str,
    lora_strength: float = 1.0,
    negative: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 97,
    frame_rate: float = 24.0,
    steps: int = 8,
    cfg: float = 3.0,
    seed: int = 1234,
    image_path: str | None = None,
    image_strength: float = 1.0,
    log_cb=None,
) -> str:
    image_filename = None
    if image_path:
        staged = backend.stage_input_image(image_path)
        image_filename = os.path.basename(staged)

    run_id = uuid.uuid4().hex[:12]
    pos_filename = f"lora_test_pos_{run_id}"
    neg_filename = f"lora_test_neg_{run_id}"
    embeddings_dir = os.path.join(backend.COMFY_ROOT, "models", "embeddings")
    pos_path = os.path.join(embeddings_dir, f"{pos_filename}.safetensors")
    neg_path = os.path.join(embeddings_dir, f"{neg_filename}.safetensors")

    try:
        _encode_prompts_native(
            prompt=prompt, negative=negative,
            pos_path=pos_path, neg_path=neg_path, log_cb=log_cb,
        )
        if not (os.path.exists(pos_path) and os.path.exists(neg_path)):
            raise RuntimeError(
                f"Etapa 1 (texto, rota nativa) terminou mas não gravou o "
                f"conditioning esperado em {embeddings_dir}."
            )

        stage2 = build_stage2_video(
            lora_name=lora_name, lora_strength=lora_strength,
            pos_filename=pos_filename, neg_filename=neg_filename,
            width=width, height=height, num_frames=num_frames, frame_rate=frame_rate,
            steps=steps, cfg=cfg, seed=seed,
            image_filename=image_filename, image_strength=image_strength,
        )
        files = backend.submit_and_wait(stage2, log_cb=log_cb, expect_node=SAVE_NODE_ID)
        if not files:
            raise RuntimeError("ComfyUI terminou sem produzir nenhum arquivo de saída.")
        src = os.path.join(backend.COMFY_ROOT, "output", files[0])
        os.makedirs(output_dir, exist_ok=True)
        dest = os.path.join(output_dir, os.path.basename(files[0]))
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.copy2(src, dest)
        return dest
    finally:
        for p in (pos_path, neg_path):
            try:
                os.remove(p)
            except OSError:
                pass
