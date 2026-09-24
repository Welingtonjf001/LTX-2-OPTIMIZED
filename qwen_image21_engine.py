"""Seletor do backend do Qwen-Image-2.1 usado pela decupagem (`QWEN21_ENGINE`).

* ``comfy`` (padrao desde 2026-09-21): ComfyUI + GGUF Q4_K_M na porta 8193 -- 5 referencias @960x544 em
  113 s / 18 GB, contra travamento do servidor diffusers (MEMORIAL 3.101).
* ``diffusers``: servidor antigo da porta 8192 (cpu-offload; ate 3 referencias).

`generate_rgba` continua no diffusers: o SaveImage do ComfyUI nao foi validado com alfa.
Mesma interface do `qwen_image21_backend`, para os pontos de chamada trocarem so o import."""
from __future__ import annotations

import os
from pathlib import Path


from qwen_image21_backend import TURNAROUND_PROMPT  # noqa: E402,F401


def _use_comfy() -> bool:
    return os.environ.get("QWEN21_ENGINE", "comfy").strip().lower() != "diffusers"


def _mod():
    if _use_comfy():
        import qwen_image21_comfy_backend as m
    else:
        import qwen_image21_backend as m
    return m


def active_port() -> int:
    m = _mod()
    return m.QWEN_COMFY_PORT if _use_comfy() else m.QWEN_IMAGE21_PORT


def all_ports() -> list[int]:
    """As duas portas: quem libera VRAM para FLUX/LTX/gate derruba qualquer uma que esteja de pe."""
    import qwen_image21_backend as old
    import qwen_image21_comfy_backend as new
    return [new.QWEN_COMFY_PORT, old.QWEN_IMAGE21_PORT]


def generate(prompt: str, out_path: Path, **kw) -> bool:
    return _mod().generate(prompt, out_path, **kw)


def edit(image_path, instruction: str, out_path: Path, **kw) -> bool:
    if _use_comfy():
        for k in ("width", "height"):   # o tamanho de saida herda o da imagem 1
            kw.pop(k, None)
    return _mod().edit(image_path, instruction, out_path, **kw)


def generate_rgba(prompt: str, out_path: Path, **kw) -> bool:
    import qwen_image21_backend as old
    return old.generate_rgba(prompt, out_path, **kw)


def shutdown_server() -> None:
    _mod().shutdown_server()
