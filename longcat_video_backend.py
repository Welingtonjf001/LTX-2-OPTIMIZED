# -*- coding: utf-8 -*-
"""LongCat-Video-Avatar via um ComfyUI SEPARADO do LTX/MiniMax H3.

Mesmo padrao de `ltx25_backend.py`/`minimax_h3_backend.py` -- dirige por HTTP
uma instalacao de ComfyUI independente (outro venv, outra porta: 8190),
disputando a MESMA 3090 fisica -- nunca rode junto com os outros dois
(ver `ensure_stopped()`/CLAUDE.md).

VALIDADO 2026-09-11 (ver memoria de projeto "LongCat-Video-Avatar validado
via ComfyUI-WanVideoWrapper"): o caminho Python direto oficial (checkpoint
fp8 comunitario via `optimum-quanto`) e' INVIAVEL nesta 3090 -- Ampere nao
tem tensor cores FP8 nativos, e o dispatch generico do quanto mede
4h23min por PASSO de amostragem. O caminho que funciona e' bf16 +
`WanVideoBlockSwap` (offload nativo do ComfyUI-WanVideoWrapper, Kijai) --
o proprio autor recomenda isso: "LongCat models only run with bf16 base
precision. Also sageattention 1.0.6 does NOT work".

Arquitetura do checkpoint: `Avatar_comfy_bf16.safetensors` e' wav2vec2/
multitalk (NAO whisper -- `LongCatAvatarWhisperEmbeds` da erro de shape).
Cinco bugs de terceiros foram contornados por fora (nenhum arquivo
vendorizado foi editado) -- ver a memoria de projeto pro detalhe de cada
um; o mais serio foi `transformers==5.x` quebrar silenciosamente
`output_hidden_states` no wav2vec2 bundled (fixado com `transformers<5`).

CLI:
    .venv\\Scripts\\python.exe -m longcat_video_backend \\
        --prompt "..." --image still.png --audio fala.wav \\
        --output-path out.mp4 --num-frames 93
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
LONGCAT_ROOT = os.environ.get("LONGCAT_VIDEO_ROOT",
                               r"E:\Users\home\Documents\LongCat-Video")
COMFY_ROOT = os.path.join(LONGCAT_ROOT, "ComfyUI")
COMFY_PORT = int(os.environ.get("LONGCAT_VIDEO_PORT", "8190"))
COMFY_SERVER = f"http://127.0.0.1:{COMFY_PORT}"
COMFY_OUTPUT = os.path.join(COMFY_ROOT, "output")

DIT_MODEL = "Avatar\\LongCat-Avatar_comfy_bf16.safetensors"
VAE_MODEL = "Wan2_1_VAE_bf16.safetensors"
TEXT_ENCODER_MODEL = "umt5-xxl-enc-bf16.safetensors"
WAV2VEC_MODEL = "wav2vec2-chinese-base_fp16.safetensors"

# Avatar 1.5 (2026-09-13): checkpoint convertido pelo Kijai (Kijai/WanVideo_comfy/LongCat),
# audio por Whisper-large-v3 em vez de wav2vec2 e LoRA DMD2 de destilacao. Ajustes do
# exemplo do proprio wrapper para LoRA de destilacao: scheduler longcat_distill_euler,
# 12 passos, shift 12, CFG 1, LoRA 0,9 sem merge. `LONGCAT_VARIANT=1.0` volta ao antigo.
VARIANT = os.environ.get("LONGCAT_VARIANT", "1.5")
DIT_MODEL_15 = "Avatar\\LongCat-Avatar-15_bf16.safetensors"
DMD_LORA_15 = "LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors"
WHISPER_MODEL = "HuMo\\whisper_large_v3_encoder_fp16.safetensors"  # nome como o /object_info lista
DEFAULTS = {"1.0": {"steps": 20, "cfg": 3.0, "scheduler": "unipc", "shift": 5.0},
            "1.5": {"steps": 12, "cfg": 1.0, "scheduler": "longcat_distill_euler", "shift": 12.0}}
BLOCKS_TO_SWAP = 25  # LongCat-Video tem 48 blocos; 25 offloaded cabe na 3090 (24GB)
# MEDIDO 2026-09-13: Avatar 1.5 + LoRA DMD sem merge, 125 quadros, 25 blocos em swap =
# 23,7 GB dedicados + 1,4 GB transbordando para a memoria compartilhada do WDDM, e o 1o
# passo passou de 18 min (o 1.0 fazia ~270 s/passo). A LoRA nao fundida carrega os pesos
# dela na GPU a cada bloco. `LONGCAT_BLOCKS_TO_SWAP` / `--blocks-to-swap` sobrepoem.
BLOCKS_TO_SWAP_15 = 36

DEFAULT_NEGATIVE_PROMPT = (
    "Close-up, bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)


def _log(msg: str, log_cb=None) -> None:
    if log_cb:
        log_cb(msg)
    else:
        print(msg, flush=True)


def server_is_up() -> bool:
    try:
        urllib.request.urlopen(f"{COMFY_SERVER}/", timeout=3)
        return True
    except Exception:
        return False


def stop_server(*, log_cb=None) -> bool:
    """Mesmo mecanismo de `generate_storyboards.stop_comfyui` -- netstat+taskkill
    pela porta, sem dependencia nova. Chame antes de subir o LTX/MiniMax H3."""
    from script_pipeline import gpu_watchdog
    return gpu_watchdog.free_port(COMFY_PORT, log=lambda m: _log(f"[longcat] {m}", log_cb))


def ensure_server(*, log_cb=None, boot_timeout: int = 180) -> None:
    """Sobe o ComfyUI do LongCat se nao estiver no ar. Idempotente."""
    if server_is_up():
        return
    python_exe = os.path.join(LONGCAT_ROOT, ".venv", "Scripts", "python.exe")
    if not os.path.isfile(python_exe):
        raise RuntimeError(f"venv do LongCat-Video nao encontrado em {python_exe}")

    _log("[longcat] ComfyUI nao esta no ar; iniciando servidor...", log_cb)
    env = os.environ.copy()
    # Mesma GPU fisica que LTX/MiniMax H3 usam (a 3090) -- ver CLAUDE.md:
    # nvidia-smi enumera ela como 0, torch/CUDA_VISIBLE_DEVICES como 1.
    env["CUDA_VISIBLE_DEVICES"] = "1"
    log_path = os.path.join(LONGCAT_ROOT, "comfyui_server.log")
    log_handle = open(log_path, "a", encoding="utf-8")
    subprocess.Popen(
        [python_exe, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT),
         "--cuda-device", "1", "--disable-auto-launch"],
        cwd=COMFY_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
    )
    inicio = time.time()
    while time.time() - inicio < boot_timeout:
        if server_is_up():
            _log(f"[longcat] ComfyUI pronto em {time.time()-inicio:.0f}s.", log_cb)
            return
        time.sleep(2)
    raise RuntimeError(f"ComfyUI do LongCat nao respondeu em {boot_timeout}s (log: {log_path})")


def _stage_input(src: str, subdir: str) -> str:
    dest_dir = os.path.join(COMFY_ROOT, "input")
    os.makedirs(dest_dir, exist_ok=True)
    dest_name = f"{subdir}_{uuid.uuid4().hex[:8]}_{os.path.basename(src)}"
    shutil.copy2(src, os.path.join(dest_dir, dest_name))
    return dest_name


def _build_workflow(*, image_name: str, audio_name: str, prompt: str, negative_prompt: str,
                     num_frames: int, steps: int, cfg: float, seed: int, fps: float,
                     variant: str = VARIANT, scheduler: str | None = None,
                     shift: float | None = None, lora_strength: float = 0.9) -> dict:
    v15 = variant == "1.5"
    base = DEFAULTS["1.5" if v15 else "1.0"]
    wf = _build_workflow_10(image_name=image_name, audio_name=audio_name, prompt=prompt,
                            negative_prompt=negative_prompt, num_frames=num_frames, steps=steps,
                            cfg=cfg, seed=seed, fps=fps)
    wf["scheduler"]["inputs"].update({"scheduler": scheduler or base["scheduler"],
                                      "shift": base["shift"] if shift is None else shift})
    blocos = os.environ.get("LONGCAT_BLOCKS_TO_SWAP")
    if blocos:
        wf["blockswap"]["inputs"]["blocks_to_swap"] = int(blocos)
    if not v15:
        return wf
    if not blocos:
        wf["blockswap"]["inputs"]["blocks_to_swap"] = BLOCKS_TO_SWAP_15
    wf["wm_model"]["inputs"]["model"] = DIT_MODEL_15
    wf["dmd_lora"] = {"class_type": "WanVideoLoraSelect", "inputs": {
        "lora": DMD_LORA_15, "strength": lora_strength,
        # merge desligado como no exemplo do wrapper: com 25 blocos em swap, fundir 1,26 GB
        # de LoRA em bf16 pede a copia inteira dos pesos na RAM.
        "low_mem_load": False, "merge_loras": False}}
    wf["wm_model"]["inputs"]["lora"] = ["dmd_lora", 0]
    # Audio do 1.5 e Whisper [T, 5, 1280]; o wav2vec2 do 1.0 da erro de shape no projetor.
    del wf["wav2vec_model"]
    wf["whisper_model"] = {"class_type": "WhisperModelLoader", "inputs": {
        "model": WHISPER_MODEL, "base_precision": "fp16", "load_device": "main_device"}}
    wf["audio_embeds"] = {"class_type": "LongCatAvatarWhisperEmbeds", "inputs": {
        "whisper_model": ["whisper_model", 0], "audio_1": ["load_audio", 0],
        "normalize_loudness": True, "num_frames": num_frames, "fps": fps,
        "audio_scale": 1.0, "audio_cfg_scale": 1.0, "multi_audio_type": "para"}}
    return wf


def _build_workflow_10(*, image_name: str, audio_name: str, prompt: str, negative_prompt: str,
                        num_frames: int, steps: int, cfg: float, seed: int, fps: float) -> dict:
    return {
        "wm_model": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": DIT_MODEL, "base_precision": "bf16", "quantization": "disabled",
                "load_device": "offload_device", "attention_mode": "sdpa",
                "block_swap_args": ["blockswap", 0],
            },
        },
        "blockswap": {
            "class_type": "WanVideoBlockSwap",
            "inputs": {
                "blocks_to_swap": BLOCKS_TO_SWAP, "offload_img_emb": False,
                "offload_txt_emb": False,
                # vace_blocks_to_swap tem que vir explicito -- bug do node:
                # fica None por padrao e `block_swap()` faz "> 0" sem checar.
                "vace_blocks_to_swap": 0,
            },
        },
        "vae": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": VAE_MODEL, "precision": "bf16"}},
        "load_image": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "text_encode": {
            "class_type": "WanVideoTextEncodeCached",
            "inputs": {
                "model_name": TEXT_ENCODER_MODEL, "precision": "bf16",
                "positive_prompt": prompt, "negative_prompt": negative_prompt,
                "quantization": "disabled", "use_disk_cache": True, "device": "gpu",
            },
        },
        "encode_image": {
            "class_type": "WanVideoEncode",
            "inputs": {
                "vae": ["vae", 0], "image": ["load_image", 0], "enable_vae_tiling": False,
                "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128,
            },
        },
        "wav2vec_model": {
            "class_type": "Wav2VecModelLoader",
            "inputs": {"model": WAV2VEC_MODEL, "base_precision": "fp16", "load_device": "main_device"},
        },
        "load_audio": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
        "audio_embeds": {
            # Arquitetura do checkpoint e' wav2vec2/multitalk, nao whisper
            # (LongCatAvatarWhisperEmbeds da' erro de shape -- ver docstring).
            "class_type": "MultiTalkWav2VecEmbeds",
            "inputs": {
                "wav2vec_model": ["wav2vec_model", 0], "audio_1": ["load_audio", 0],
                "normalize_loudness": True, "num_frames": num_frames, "fps": fps,
                "audio_scale": 1.0, "audio_cfg_scale": 1.0, "multi_audio_type": "para",
            },
        },
        "extend_embeds": {
            "class_type": "WanVideoLongCatAvatarExtendEmbeds",
            "inputs": {
                "prev_latents": ["encode_image", 0], "audio_embeds": ["audio_embeds", 0],
                # overlap=0 crasha o node quando ref_latent tambem e' passado
                # (bug: "latent_overlap referenced before assignment"); overlap=1
                # tambem e' o correto pra I2V (ancora o frame inicial na imagem).
                "num_frames": num_frames, "overlap": 1, "frames_processed": 0,
                "if_not_enough_audio": "pad_with_start",
                "ref_frame_index": 10, "ref_mask_frame_range": 3,
                "ref_latent": ["encode_image", 0],
            },
        },
        "scheduler": {
            "class_type": "WanVideoSchedulerv2",
            "inputs": {"scheduler": "unipc", "steps": steps, "shift": 5.0, "start_step": 0, "end_step": -1},
        },
        "sampler": {
            "class_type": "WanVideoSamplerv2",
            "inputs": {
                "model": ["wm_model", 0], "image_embeds": ["extend_embeds", 0],
                "cfg": cfg, "seed": seed, "force_offload": True,
                "scheduler": ["scheduler", 0], "text_embeds": ["text_encode", 0],
            },
        },
        "decode": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "vae": ["vae", 0], "samples": ["sampler", 0], "enable_vae_tiling": False,
                "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128,
            },
        },
        "save": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["decode", 0], "frame_rate": fps, "loop_count": 0,
                "filename_prefix": "LongCatAvatar", "format": "video/h264-mp4",
                "pingpong": False, "save_output": True, "audio": ["load_audio", 0],
                "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True,
            },
        },
    }


def _submit_and_wait(workflow: dict, *, log_cb=None, timeout: int) -> dict:
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_SERVER}/prompt", data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejeitou o workflow (HTTP {e.code}): {body[:3000]}")

    prompt_id = result["prompt_id"]
    _log(f"[longcat] enfileirado: {prompt_id}", log_cb)

    inicio = time.time()
    while time.time() - inicio < timeout:
        with urllib.request.urlopen(f"{COMFY_SERVER}/history/{prompt_id}", timeout=30) as resp:
            history = json.loads(resp.read())
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            if status.get("status_str") == "error":
                for m in status.get("messages", []):
                    if m[0] == "execution_error":
                        tb = "\n".join(m[1].get("traceback", []))
                        raise RuntimeError(f"{m[1].get('node_type')}: {m[1].get('exception_message')}\n{tb}")
                raise RuntimeError(f"ComfyUI reportou erro: {json.dumps(status, ensure_ascii=False)}")
        time.sleep(5)
        _log(f"[longcat] gerando... {time.time()-inicio:.0f}s", log_cb)
    raise TimeoutError(f"LongCat nao terminou em {timeout}s")


def generate(prompt: str, output_path: str, *, image_path: str, audio_path: str,
             negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
             num_frames: int = 93, steps: int | None = None, cfg: float | None = None,
             seed: int = 42, fps: float = 25.0, log_cb=None, timeout: int | None = None,
             variant: str = VARIANT, lora_strength: float = 0.9) -> str:
    """Gera um clipe LongCat-Video-Avatar (imagem + audio + texto) e copia para
    *output_path*. `image_path` e' a referencia de identidade/still do plano;
    `audio_path` e' a fala real que dirige o lip-sync (mesmo papel do
    `audio_conditioning` do LTX 2.5 -- NAO opcional aqui, o modelo e' avatar
    audio-driven, sem audio nao ha o que sincronizar).

    `timeout` None = estimado. MEDIDO 2026-09-13 na 3090 (bf16, 25 blocos em swap,
    960x544): 101 quadros a ~270 s/passo -- 20 passos = ~90 min. O fixo de 3600s
    abandonava o job no passo 13 com o servidor ainda gerando."""
    base = DEFAULTS["1.5" if variant == "1.5" else "1.0"]
    steps = base["steps"] if steps is None else steps
    cfg = base["cfg"] if cfg is None else cfg
    if timeout is None:
        # 270 s/passo medido no 1.0 com CFG 3 (dois passes por passo); CFG 1 faz um so.
        por_passo = 270 * (1.0 if cfg != 1.0 else 0.6)
        timeout = int(900 + steps * por_passo * max(1.0, num_frames / 101) * 1.5)
    ensure_server(log_cb=log_cb)
    image_name = _stage_input(image_path, "img")
    audio_name = _stage_input(audio_path, "aud")
    workflow = _build_workflow(
        image_name=image_name, audio_name=audio_name, prompt=prompt,
        negative_prompt=negative_prompt, num_frames=num_frames, steps=steps,
        cfg=cfg, seed=seed, fps=fps, variant=variant, lora_strength=lora_strength,
    )
    _log(f"[longcat] Avatar {variant}: {steps} passos, CFG {cfg:g}, timeout {timeout}s", log_cb)
    entry = _submit_and_wait(workflow, log_cb=log_cb, timeout=timeout)
    outputs = entry.get("outputs", {}).get("save", {})
    videos = outputs.get("gifs") or outputs.get("videos") or []
    if not videos:
        raise RuntimeError(f"LongCat concluiu mas nenhum video foi encontrado: {outputs}")
    v = videos[0]
    src = os.path.join(COMFY_OUTPUT, v.get("subfolder", ""), v["filename"])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    shutil.copy2(src, output_path)
    _log(f"[longcat] salvo em {output_path}", log_cb)
    return output_path


def _cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--num-frames", type=int, default=93)
    ap.add_argument("--variant", default=VARIANT, choices=["1.0", "1.5"],
                    help="1.5 = Whisper + LoRA DMD (padrao); 1.0 = wav2vec2, 20 passos, CFG 3")
    ap.add_argument("--lora-strength", type=float, default=0.9, help="LoRA DMD do 1.5")
    ap.add_argument("--steps", type=int, default=None, help="padrao por variante (1.5: 12, 1.0: 20)")
    ap.add_argument("--cfg", type=float, default=None, help="padrao por variante (1.5: 1, 1.0: 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--timeout", type=int, default=None, help="segundos; padrao estimado por quadros x passos")
    args = ap.parse_args(argv)

    generate(args.prompt, args.output_path, image_path=args.image, audio_path=args.audio,
             num_frames=args.num_frames, steps=args.steps, cfg=args.cfg, seed=args.seed, fps=args.fps,
             timeout=args.timeout, variant=args.variant, lora_strength=args.lora_strength)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
