"""HTTP bridge to the separately installed Qwen-Image-2.1 service.

The model lives in ``E:\\Users\\home\\Documents\\Qwen-Image-2.1`` with its own
Torch/Diffusers environment. This module deliberately communicates over HTTP so
the LTX virtualenv never imports its dependency stack.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


QWEN_IMAGE21_ROOT = Path(r"E:\Users\home\Documents\Qwen-Image-2.1")
QWEN_IMAGE21_PORT = 8192
QWEN_IMAGE21_URL = f"http://127.0.0.1:{QWEN_IMAGE21_PORT}"
_server_proc = None


def server_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{QWEN_IMAGE21_URL}/health", timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def ensure_server(log_cb=None, boot_timeout: int = 60) -> None:
    global _server_proc
    log = log_cb or print
    if server_is_up():
        return
    python_exe = QWEN_IMAGE21_ROOT / ".venv" / "Scripts" / "python.exe"
    server = QWEN_IMAGE21_ROOT / "server.py"
    if not python_exe.is_file() or not server.is_file():
        raise RuntimeError(f"Qwen-Image-2.1 is not installed at {QWEN_IMAGE21_ROOT}")
    logs_dir = QWEN_IMAGE21_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    handle = open(logs_dir / "server.log", "a", encoding="utf-8", buffering=1)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log("[qwen-image-2.1] servidor nao esta respondendo; iniciando...")
    _server_proc = subprocess.Popen([str(python_exe), str(server), "--port", str(QWEN_IMAGE21_PORT)],
                                    cwd=str(QWEN_IMAGE21_ROOT), env=env,
                                    stdout=handle, stderr=subprocess.STDOUT)
    started = time.monotonic()
    while time.monotonic() - started < boot_timeout:
        if server_is_up():
            log("[qwen-image-2.1] servidor pronto; o primeiro still carrega os pesos.")
            return
        time.sleep(1)
    raise RuntimeError(f"Qwen-Image-2.1 did not start; inspect {logs_dir / 'server.log'}")


# MEDIDO 2026-09-21 (3090, cpu-offload, 960x544): 2 refs 43 s, 3 refs 52 s, 4 refs a 640x352 287 s,
# 4 refs a 960x544 > 600 s, 5 refs > 4 h (24 GB de VRAM a 100%, cada passo mais lento que o anterior).
# O servidor atende um pedido por vez e NAO cancela: uma geracao presa bloqueia tudo que vier depois.
MAX_REFERENCES = 3
DEFAULT_TIMEOUT = 420  # ~10x um still normal (~40 s); 1800 s deixava a corrida parada 30 min por still preso


def _reference_bytes(path: Path) -> bytes:
    """Bytes da referencia; PNG com transparencia e achatado sobre cinza neutro.

    O servidor faz `.convert("RGB")`, que descarta o canal alfa e deixa nos pixels transparentes
    o RGB que estava por baixo (preto/lixo). Um recorte RGBA (aeronave, sujeito extraido) entraria
    como referencia com fundo sujo."""
    raw = Path(path).read_bytes()
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        if im.mode in ("RGBA", "LA") or "transparency" in im.info:
            base = Image.new("RGBA", im.size, (127, 127, 127, 255))
            base.alpha_composite(im.convert("RGBA"))
            out = io.BytesIO()
            base.convert("RGB").save(out, format="PNG")
            return out.getvalue()
    except Exception:  # noqa: BLE001 - qualquer coisa que nao seja imagem segue como veio
        pass
    return raw


def generate(prompt: str, out_path: Path, *, width: int = 1024, height: int = 1024,
             steps: int = 30, seed: int = 42, reference_images: list[str] | None = None,
             reference_roles: list[str] | None = None,
             timeout: int = DEFAULT_TIMEOUT, log=print) -> bool:
    """`reference_roles`: mesmo papel nomeado do backend ComfyUI
    (`qwen_image21_comfy_backend.roles_prefix`) -- aqui não há workflow com
    campo de imagem dedicado por referência, então o papel entra como texto
    no PRÓPRIO prompt (`<image1> is X.`), na esperança de que o servidor
    HTTP local trate a ordem das referências do mesmo jeito. Não testado
    contra este backend especificamente -- só contra o ComfyUI."""
    ensure_server(log)
    if reference_roles:
        prompt = " ".join(f"<image{i}> is {r.rstrip('. ')}." for i, r in enumerate(reference_roles, 1) if r) \
            + " " + prompt
    refs = [Path(p) for p in (reference_images or []) if p]
    if len(refs) > MAX_REFERENCES:
        log(f"[qwen-image-2.1] AVISO: {len(refs)} referencias; usando as {MAX_REFERENCES} primeiras "
            "(mais que isso estoura a VRAM da 3090 com offload -- ver MAX_REFERENCES).")
        refs = refs[:MAX_REFERENCES]
    payload = {
        "prompt": prompt, "width": int(width), "height": int(height),
        "steps": int(steps), "seed": int(seed),
        "reference_b64s": [base64.b64encode(_reference_bytes(path)).decode("ascii") for path in refs],
    }
    request = urllib.request.Request(
        f"{QWEN_IMAGE21_URL}/generate", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        log(f"[qwen-image-2.1] geracao falhou: {type(exc).__name__}: {exc}")
        if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
            # A geracao continua no servidor (nao ha cancelamento) e segura o lock: reinicia o
            # servidor, senao TODO pedido seguinte espera na fila ate o proprio timeout.
            try:
                from script_pipeline import gpu_watchdog
                gpu_watchdog.free_port(QWEN_IMAGE21_PORT, log=log)
                log("[qwen-image-2.1] servidor reiniciado apos timeout (o proximo pedido o sobe de novo).")
            except Exception as kill_exc:  # noqa: BLE001
                log(f"[qwen-image-2.1] nao consegui reiniciar o servidor: {kill_exc}")
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(body["image_b64"]))
    log(f"[qwen-image-2.1] still salvo em {out_path} ({body.get('elapsed_seconds', 0):.1f}s)")
    return True


def shutdown_server() -> None:
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
    _server_proc = None


def _round16(value: int) -> int:
    return max(256, min(2048, int(value) // 16 * 16))


def edit(image_path: str | Path, instruction: str, out_path: Path, *, steps: int = 30, seed: int = 42,
         extra_references: list[str] | None = None, width: int | None = None, height: int | None = None,
         timeout: int = DEFAULT_TIMEOUT, log=print) -> bool:
    """Edicao nativa: a imagem entra como PRIMEIRA referencia e o prompt e a instrucao.

    MEDIDO 2026-09-21 (mesmo servidor): mudar a expressao de dois personagens mantendo cabine,
    uniformes e identidade (ArcFace 0,54/0,53) em 39 s. O tamanho de saida segue o da imagem."""
    from PIL import Image
    with Image.open(image_path) as im:
        w, h = im.size
    return generate(instruction, out_path, width=width or _round16(w), height=height or _round16(h),
                    steps=steps, seed=seed, reference_images=[str(image_path), *(extra_references or [])],
                    timeout=timeout, log=log)


RGBA_TEMPLATE = ("This is an RGBA image with transparency. {prompt}. "
                 "The image has alpha channel and the background is transparent.")


def generate_rgba(prompt: str, out_path: Path, *, width: int = 1024, height: int = 576, steps: int = 30,
                  seed: int = 42, reference_images: list[str] | None = None, min_transparent: float = 0.05,
                  timeout: int = DEFAULT_TIMEOUT, log=print) -> bool:
    """Recorte com alfa nativo (aeronave, props, sujeito). Confere que o alfa e de verdade."""
    ok = generate(RGBA_TEMPLATE.format(prompt=prompt.rstrip(". ")), out_path, width=width, height=height,
                  steps=steps, seed=seed, reference_images=reference_images, timeout=timeout, log=log)
    if not ok:
        return False
    import numpy as np
    from PIL import Image
    with Image.open(out_path) as im:
        if im.mode != "RGBA":
            log(f"[qwen-image-2.1] AVISO: saida {im.mode}, sem alfa: {out_path}")
            return False
        frac = float((np.asarray(im)[..., 3] < 16).mean())
    if frac < min_transparent:
        log(f"[qwen-image-2.1] AVISO: so {frac:.1%} transparente (minimo {min_transparent:.0%}): {out_path}")
        return False
    return True


TURNAROUND_PROMPT = ("Character turnaround sheet of the person in the reference photo, wearing {outfit}: front view, "
                     "three-quarter view and back view side by side, full body, neutral pose, plain white background, "
                     "consistent face, hairstyle and outfit in all three views.")

