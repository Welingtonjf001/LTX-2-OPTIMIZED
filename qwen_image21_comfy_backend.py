"""Qwen-Image-2.1 pelo ComfyUI (GGUF), porta 8193 -- mesma interface do `qwen_image21_backend`.

Instalacao separada em ``E:\\Users\\home\\Documents\\Qwen-Image-2.1\\ComfyUI`` (ComfyUI upstream +
ComfyUI-GGUF do leejet, venv proprio). Diferencas para o servidor diffusers (porta 8192):

* transformer em GGUF (Q4_K_M ~4,6 GB, Q8_0 ~7,6 GB) e encoder int8 (~9,4 GB) -- cabem na 3090
  sem o `enable_model_cpu_offload` que estourava a VRAM com 4-5 referencias;
* as referencias sao citadas no prompt como ``<image1>``..``<image10>`` (sintaxe nativa do
  ``TextEncodeQwenImage21``); ``reference_roles`` monta essa frase para amarrar cada foto a um papel;
* ``ref_resolution`` = orcamento de pixels POR REFERENCIA (o codificador as redimensiona para cerca
  de resolution x resolution) -- e o botao que controla VRAM/tempo com muitas referencias;
* cancelamento real (``/interrupt``): uma geracao presa nao segura mais a fila.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

QWEN_COMFY_ROOT = Path(r"E:\Users\home\Documents\Qwen-Image-2.1\ComfyUI")
QWEN_COMFY_PORT = 8193
QWEN_COMFY_URL = f"http://127.0.0.1:{QWEN_COMFY_PORT}"
UNET = os.environ.get("QWEN21_GGUF", "qwen-image-2.1-Q4_K_M.gguf")
CLIP = "qwen3vl_8b_int8_convrot.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"
DEFAULT_TIMEOUT = 420
MAX_REFERENCES = 10  # limite nativo do modelo; o custo real esta em ref_resolution (ver MEMORIAL 3.101)
_server_proc = None


def _http(path: str, payload: dict | None = None, timeout: int = 30):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(f"{QWEN_COMFY_URL}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def server_is_up() -> bool:
    try:
        _http("/system_stats", timeout=3)
        return True
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


def ensure_server(log_cb=None, boot_timeout: int = 180) -> None:
    global _server_proc
    log = log_cb or print
    if server_is_up():
        return
    python_exe = QWEN_COMFY_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python_exe.is_file():
        raise RuntimeError(f"ComfyUI do Qwen-Image-2.1 nao instalado em {QWEN_COMFY_ROOT}")
    logs = QWEN_COMFY_ROOT.parent / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0", PYTHONUTF8="1")
    log("[qwen-comfy] servidor nao esta respondendo; iniciando...")
    _server_proc = subprocess.Popen(
        [str(python_exe), "main.py", "--listen", "127.0.0.1", "--port", str(QWEN_COMFY_PORT), "--disable-auto-launch"],
        cwd=str(QWEN_COMFY_ROOT), env=env, stdout=open(logs / "comfyui_qwen.log", "a", encoding="utf-8", buffering=1),
        stderr=subprocess.STDOUT)
    started = time.monotonic()
    while time.monotonic() - started < boot_timeout:
        if server_is_up():
            return
        time.sleep(2)
    raise RuntimeError(f"ComfyUI do Qwen nao subiu; veja {logs / 'comfyui_qwen.log'}")


def roles_prefix(roles: list[str]) -> str:
    """"<image1> is X. <image2> is Y." -- amarra cada referencia a um papel (evita colapso de identidades)."""
    return " ".join(f"<image{i}> is {role.rstrip('. ')}." for i, role in enumerate(roles, 1) if role) + " "


def build_workflow(prompt: str, image_names: list[str], *, width: int, height: int, steps: int, seed: int,
                   ref_resolution: int = 1024, edit_target: bool = False, cache_device: str = "auto",
                   cache_dtype: str = "default", unet: str | None = None, prefix: str = "qwen21") -> dict:
    """Grafo API: GGUF + QwenImage21Cache + TextEncodeQwenImage21 (+ refs) + KSampler euler/simple cfg 1.

    edit_target=True: a imagem 1 e o alvo de edicao e a saida herda o tamanho dela (latente do codificador);
    False: canvas vazio width x height e todas as imagens sao apenas referencias."""
    g: dict = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet or UNET}},
        "2": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": cache_device, "dtype": cache_dtype}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }
    enc_inputs = {"clip": ["3", 0], "prompt": prompt, "negative_prompt": "", "resolution": int(ref_resolution),
                  "vae": ["4", 0]}
    for i, name in enumerate(image_names, 1):
        g[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        enc_inputs[f"images.image_{i}"] = [f"img{i}", 0]
    g["5"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc_inputs}
    if edit_target and image_names:
        latent = ["5", 2]
    else:
        g["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": int(width), "height": int(height), "batch_size": 1}}
        latent = ["6", 0]
    g["7"] = {"class_type": "KSampler", "inputs": {
        "model": ["2", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": latent, "seed": int(seed),
        "steps": int(steps), "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["4", 0]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return g


def _stage(path: str | Path) -> str:
    """Copia a referencia (achatada sobre cinza se tiver alfa) para ComfyUI/input com nome unico."""
    from qwen_image21_backend import _reference_bytes
    inp = QWEN_COMFY_ROOT / "input"
    inp.mkdir(exist_ok=True)
    name = f"q21_{uuid.uuid4().hex[:8]}_{Path(path).stem}.png"
    (inp / name).write_bytes(_reference_bytes(Path(path)))
    return name


def interrupt() -> None:
    try:
        _http("/interrupt", {})
        _http("/queue", {"clear": True})
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        pass


def generate(prompt: str, out_path: Path, *, width: int = 1024, height: int = 1024, steps: int = 30, seed: int = 42,
             reference_images: list[str] | None = None, reference_roles: list[str] | None = None,
             ref_resolution: int = 1024, edit_target: bool = False, timeout: int = DEFAULT_TIMEOUT,
             cache_device: str = "auto", cache_dtype: str = "default", unet: str | None = None, log=print) -> bool:
    ensure_server(log)
    refs = [Path(p) for p in (reference_images or []) if p][:MAX_REFERENCES]
    if reference_roles:
        prompt = roles_prefix(reference_roles) + prompt
    staged = [_stage(p) for p in refs]
    try:
        return _generate_staged(prompt, out_path, staged, width=width, height=height, steps=steps, seed=seed,
                                ref_resolution=ref_resolution, edit_target=edit_target, timeout=timeout,
                                cache_device=cache_device, cache_dtype=cache_dtype, unet=unet, log=log)
    finally:
        # As referencias staged em ComfyUI/input nunca eram removidas -- numa decupagem de
        # dezenas de planos isso acumulava PNGs orfaos sem limite (achado da auditoria 2026-09-21).
        for name in staged:
            (QWEN_COMFY_ROOT / "input" / name).unlink(missing_ok=True)


def _generate_staged(prompt: str, out_path: Path, staged: list[str], *, width: int, height: int, steps: int,
                     seed: int, ref_resolution: int, edit_target: bool, timeout: int, cache_device: str,
                     cache_dtype: str, unet: str | None, log) -> bool:
    workflow = build_workflow(prompt, staged, width=_r32(width), height=_r32(height), steps=steps, seed=seed,
                              ref_resolution=ref_resolution, edit_target=edit_target, cache_device=cache_device,
                              cache_dtype=cache_dtype, unet=unet)
    started = time.monotonic()
    try:
        prompt_id = _http("/prompt", {"prompt": workflow, "client_id": uuid.uuid4().hex})["prompt_id"]
    except urllib.error.HTTPError as exc:
        log(f"[qwen-comfy] workflow recusado: {exc.read().decode('utf-8', 'replace')[:600]}")
        return False
    except (urllib.error.URLError, OSError, TimeoutError, KeyError, ValueError) as exc:
        log(f"[qwen-comfy] falha ao enviar: {type(exc).__name__}: {exc}")
        return False
    entry = None
    while time.monotonic() - started < timeout:
        try:
            hist = _http(f"/history/{prompt_id}", timeout=10)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            hist = {}
        entry = hist.get(prompt_id)
        if entry and entry.get("status", {}).get("completed") is not None and entry.get("outputs") is not None:
            break
        time.sleep(1.0)
    else:
        log(f"[qwen-comfy] timeout de {timeout}s; cancelando a geracao (interrupt).")
        interrupt()
        return False
    if entry.get("status", {}).get("status_str") == "error":
        msgs = [m for m in entry["status"].get("messages", []) if m and m[0] == "execution_error"]
        log(f"[qwen-comfy] erro de execucao: {json.dumps(msgs[-1][1], ensure_ascii=False)[:500] if msgs else entry['status']}")
        return False
    for out in (entry.get("outputs") or {}).values():
        for item in out.get("images", []) or []:
            src = QWEN_COMFY_ROOT / "output" / item.get("subfolder", "") / item["filename"]
            if src.exists():
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out_path)
                log(f"[qwen-comfy] still salvo em {out_path} ({time.monotonic() - started:.1f}s)")
                return True
    log("[qwen-comfy] concluiu mas nao achei a imagem de saida")
    return False


def _r32(value: int) -> int:
    return max(256, min(2048, int(value) // 32 * 32))


def edit(image_path: str | Path, instruction: str, out_path: Path, *, extra_references: list[str] | None = None,
         steps: int = 30, seed: int = 42, ref_resolution: int = 0, timeout: int = DEFAULT_TIMEOUT, log=print) -> bool:
    """Edicao nativa: imagem 1 = alvo (a saida herda o tamanho dela); demais = referencias (`<image2>`...)."""
    return generate(instruction, out_path, steps=steps, seed=seed, reference_images=[str(image_path), *(extra_references or [])],
                    edit_target=True, ref_resolution=ref_resolution, timeout=timeout, log=log)


def shutdown_server() -> None:
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
    _server_proc = None
