"""roop-unleashed face-swap backend: an optional post-processing step run
AFTER a scene's video is generated (by either the fp8 or GGUF backend).

roop-unleashed lives in its own fully isolated environment
(E:\\Users\\home\\Documents\\roop\\roop-unleashed\\.venv, Python 3.10, its own
torch 2.5.1+cu124 / onnxruntime-gpu / insightface build) -- a completely
different stack from this project's .venv, installed and validated in a
separate session. Rather than importing roop's modules directly (which would
require installing its heavy, version-pinned dependency set into THIS venv,
risking conflicts with the LTX/ComfyUI stack), each swap runs as a short-lived
subprocess using roop's own python.exe and roop_swap_runner.py (a thin,
argparse-driven wrapper around roop's internal
extract_face_images -> FaceSet -> batch_process_regular pipeline -- the same
calls its own Gradio "Start" button makes, bypassing that button's stateful,
interactive-only UI plumbing).

Cost: per-call model reload is CHEAP here (insightface's face-detection/swap
models are small ONNX files; ~2-3s to load, not the 22B-scale cost of the LTX
transformer), so a persistent-server pattern like gguf_backend.py's ComfyUI
process is not needed -- a fresh subprocess per swap is simple and adequate.

Validated end-to-end (2026-07-25 session): a 9-frame 1280x704 clip swapped
correctly on the RTX 3090 via this exact extract->FaceSet->batch_process
call chain, producing a valid H.264 output.
"""
import os
import subprocess
import sys

ROOP_ROOT = r"E:\Users\home\Documents\roop\roop-unleashed"
ROOP_PYTHON = os.path.join(ROOP_ROOT, ".venv", "Scripts", "python.exe")
ROOP_RUNNER = os.path.join(ROOP_ROOT, "roop_swap_runner.py")

DEFAULT_SWAP_MODEL = "InSwapper 128"
DEFAULT_CUDA_DEVICE_ID = 0  # roop's OWN device numbering, set via CUDA_VISIBLE_DEVICES below


def is_installed() -> bool:
    return os.path.exists(ROOP_PYTHON) and os.path.exists(ROOP_RUNNER)


def swap_face(
    source_face_path: str,
    target_video_path: str,
    output_path: str,
    swap_model: str = DEFAULT_SWAP_MODEL,
    enhancer: str | None = None,
    face_swap_mode: str = "first",
    cuda_visible_device: str = "1",  # physical GPU index (this machine: 1 = RTX 3090)
    log_cb=None,
    timeout: int = 600,
) -> bool:
    """Run roop's face swap as a subprocess in its own environment. Returns
    True on success, with *output_path* written on disk."""
    def log(msg):
        if log_cb:
            log_cb(msg)
        print(f"[ROOP] {msg}", flush=True)

    if not is_installed():
        log(f"ERRO: roop-unleashed não está instalado em {ROOP_ROOT} "
            "(.venv ou roop_swap_runner.py ausente).")
        return False
    if not os.path.exists(source_face_path):
        log(f"ERRO: imagem de referência não encontrada: {source_face_path}")
        return False
    if not os.path.exists(target_video_path):
        log(f"ERRO: vídeo alvo não encontrado: {target_video_path}")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_device

    cmd = [
        ROOP_PYTHON, "-u", ROOP_RUNNER,
        "--source", source_face_path,
        "--target", target_video_path,
        "--output", output_path,
        "--cuda-device-id", "0",  # index WITHIN the CUDA_VISIBLE_DEVICES-filtered view
        "--swap-model", swap_model,
        "--face-swap-mode", face_swap_mode,
    ]
    if enhancer:
        cmd.extend(["--enhancer", enhancer])

    log(f"trocando rosto: {os.path.basename(target_video_path)} <- {os.path.basename(source_face_path)} "
        f"(swap_model={swap_model}, enhancer={enhancer or 'nenhum'})")

    proc = subprocess.Popen(
        cmd, cwd=ROOP_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
    )
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        log(f"timeout após {timeout}s; processo encerrado.")
        return False

    if proc.returncode != 0:
        log(f"roop_swap_runner saiu com código {proc.returncode}.")
        return False
    if not os.path.exists(output_path):
        log("processo terminou sem erro reportado, mas o arquivo de saída não existe.")
        return False
    return True
