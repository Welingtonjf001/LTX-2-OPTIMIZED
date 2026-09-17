"""Ponte HTTP para o Z-Image-Turbo, pedido do usuario 2026-09-17 ("acople aos
arquivos... teste com imagem de referencia").

Instalacao SEPARADA em `E:\\Users\\home\\Documents\\Z-Image` (venv proprio,
diffusers puro -- sem ComfyUI), porta 8191. Mesmo padrao dos outros motores
de terceiros deste projeto (`ltx25_backend.py` porta 8188,
`minimax_h3_backend.py` porta 8189, `longcat_video_backend.py` porta 8190):
processo dirigido por HTTP, nao importado em processo (versoes de
torch/diffusers incompativeis com o `.venv` do LTX -- ver CLAUDE.md, secao de
ambientes Python). Disputa a MESMA 3090 fisica -- nao rode junto com
ComfyUI/geracao de video pesada.

Z-Image-Turbo e um modelo de IMAGEM (T2I), nao substitui LTX/MiniMax/LongCat
(video) -- e uma alternativa a FLUX/SD3.5/SDXL para os STILLS da decupagem
(`--image-engine zimage`, ver generate_storyboards.py). Com imagem de
referencia (img2img via ZImageImg2ImgPipeline, MESMOS pesos do Z-Image-Turbo,
nao precisa da variante Edit) preserva composicao/identidade e deixa o prompt
reescrever o resto -- mesmo papel que `_wire_img2img_reference` cumpre para
sd35/flux1 aqui, so que via um servidor externo em vez de um grafo ComfyUI.

MEDIDO 2026-09-17 (3090, 704x960, 9 passos): carga do pipeline ~220s (uma vez,
o servidor mantem os pesos residentes), geracao ~19s/still depois disso, pico
de VRAM 12,9 GB.
"""
from __future__ import annotations

import base64
import os
import subprocess
import time
from pathlib import Path

ZIMAGE_ROOT = Path(r"E:\Users\home\Documents\Z-Image")
ZIMAGE_PORT = 8191
ZIMAGE_URL = f"http://127.0.0.1:{ZIMAGE_PORT}"

_server_proc = None


def server_is_up() -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{ZIMAGE_URL}/health", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def ensure_server(log_cb=None, boot_timeout: int = 60) -> None:
    """Sobe o servidor do Z-Image se nao estiver no ar. Idempotente.

    `boot_timeout` cobre so o BOOT do processo HTTP (segundos), nao a carga
    do modelo -- o pipeline carrega sob demanda no primeiro /generate (lazy,
    ver server.py), e essa carga (~220s medido) e coberta pelo timeout da
    proria chamada HTTP em `generate()`."""
    global _server_proc

    def log(msg):
        (log_cb or print)(msg)

    if server_is_up():
        return
    log("[zimage] servidor nao esta respondendo; iniciando...")
    env = os.environ.copy()
    # Convencao PROPRIA deste repo (ver start_server.bat) -- NAO e a mesma do
    # LTX-2-OPTIMIZED (CUDA_VISIBLE_DEVICES=1 la). Com PCI_BUS_ID + "0" o
    # torch enumera igual ao nvidia-smi (MEDIDO): 0=3090, 1=4070.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    python_exe = str(ZIMAGE_ROOT / ".venv" / "Scripts" / "python.exe")
    logs_dir = ZIMAGE_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / "zimage_server.log"
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    _server_proc = subprocess.Popen(
        [python_exe, "server.py", "--port", str(ZIMAGE_PORT)],
        cwd=str(ZIMAGE_ROOT), env=env,
        stdout=log_handle, stderr=subprocess.STDOUT,
    )
    t0 = time.time()
    while time.time() - t0 < boot_timeout:
        if server_is_up():
            log(f"[zimage] servidor pronto ({time.time() - t0:.0f}s) -- "
                "o PRIMEIRO /generate ainda carrega o modelo (~220s medido).")
            return
        time.sleep(1)
    log(f"[zimage] AVISO: servidor nao respondeu em {boot_timeout}s -- "
        f"confira {log_path}")


def generate(prompt: str, out_path: Path, *, width: int = 960, height: int = 1024,
            steps: int = 9, seed: int = 42, reference_image: str | None = None,
            strength: float = 0.55, timeout: int = 400, log=print) -> bool:
    """Gera um still. `reference_image`: caminho local de uma imagem -- vai
    junto como base64 no corpo da requisicao (o servidor roda no MESMO
    computador, sem custo de rede real). None = texto-puro."""
    import json
    import urllib.error
    import urllib.request

    ensure_server(log_cb=log)

    payload = {"prompt": prompt, "width": width, "height": height,
               "steps": steps, "seed": seed, "strength": strength}
    if reference_image:
        ref_bytes = Path(reference_image).read_bytes()
        payload["reference_b64"] = base64.b64encode(ref_bytes).decode("ascii")

    req = urllib.request.Request(
        f"{ZIMAGE_URL}/generate", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log(f"[zimage] geracao falhou: {type(e).__name__}: {e}")
        return False

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(body["image_b64"]))
    log(f"[zimage] still salvo em {out_path} ({body.get('elapsed_seconds', 0):.1f}s)")
    return True


def shutdown_server() -> None:
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
    _server_proc = None
